import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))

import torch
import esm
from torch.nn.utils.rnn import pad_sequence

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

device = 'cpu'
model.to(device)
model.eval()  

MAX_SEQ_LEN = 2000
ESM2_EMBEDDING_LEN = 1280


def pad_collate(batch):
    # Shorten all residues above MAX_SEQ_LEN for any x above MAX_SEQ_LEN in length.
    batch = [(x[:MAX_SEQ_LEN] if len(x) > MAX_SEQ_LEN else x, y) for x, y in batch]
    # This was code to filter out sequences above MAX_SEQ_LEN
    # batch = [(x, y) for x, y in batch if len(x) <= MAX_SEQ_LEN]
    (xx, yy) = zip(*batch)
    xx = [torch.from_numpy(x).float() for x in xx]
    yy = [torch.from_numpy(y).float() for y in yy]

    x_lens = torch.Tensor([len(x) for x in xx]).float()
    y_lens = torch.Tensor([len(y) for y in yy]).float()
    assert all(x_lens <= MAX_SEQ_LEN)
    # Pad everything up to MAX_SEQ_LEN
    xx[0] = torch.nn.ConstantPad2d((0, 0, 0, MAX_SEQ_LEN - xx[0].shape[0]), 0)(xx[0])
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    return xx_pad, yy_pad, x_lens, y_lens


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def call_esm_model(data_chunk):
    batch_labels, batch_strs, batch_tokens = batch_converter(data_chunk)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    aa_reps = results["representations"][33].cpu().detach().numpy()
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    mean_rep = []
    for i, tokens_len in enumerate(batch_lens):
        mean_rep.append(aa_reps[i, 1 : tokens_len - 1].mean(0))
    return mean_rep[0], aa_reps[0]

