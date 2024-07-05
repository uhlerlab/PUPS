import os
import torch
import copy
from tqdm import tqdm
import pandas as pd
import pdb


def call_model(model, X_esm2_encoding, X_protein_len, X_landmark_stains, embedding_hooks=['joint_embedding']):
    activation = {}

    def get_activation(name, rename):
        def hook(model, input, output):
            activation[rename] = output.detach()

        return hook

    if 'protein_embedding' in embedding_hooks:
        model.light_attention_trunk.linear.register_forward_hook(
            get_activation("3", "protein_embedding")
        )
    if 'cell_embedding' in embedding_hooks:
        model.cell_inpainting_unet.bottleneck_layer1.register_forward_hook(
            get_activation("2", "cell_embedding")
        )
    if 'joint_embedding' in embedding_hooks: 
        model.cell_inpainting_unet.upsample_layer1.register_forward_hook(
            get_activation("0", "joint_embedding_after_upsample_layer1")
        )

    y_pred_antibody_stain, _y_pred_multilabel, y_pred_ranked = model.predict_step(
        (
            X_esm2_encoding.unsqueeze(0),
            torch.Tensor([X_protein_len]),
            torch.from_numpy(X_landmark_stains).unsqueeze(0),
            None,
            None,
        ),
        batch_idx=0,
    )
    return y_pred_ranked, y_pred_antibody_stain, copy.deepcopy(activation)


def process_matrix(
    model,
    output_pkl_file,
    dataset_1,
    dataset_2=None,
    dataset_split=0.5,
    total_datapoints=30,
):
    if os.path.exists(output_pkl_file):
        embedding_df = pd.read_pickle(output_pkl_file)
    else:
        embedding_df = pd.DataFrame(
            columns=[
                "isoform_idx",
                "cell_idx",
                "splice_variant_id",
                "cell_line",
                "cell_image",
                "activation",
                "cell_line_metadata",
                "isoform_metadata",
                "y_pred_multilabel",
                "y_pred_antibody_stain_pred",
                "y_antibody_stain_orig",
            ]
        )
        embedding_df.set_index(["isoform_idx", "cell_idx"], inplace=True)

    def resolve_dataset(idx, total_datapoints):
        # In the case of multiple datasets, split the calls
        if dataset_split > float(idx / total_datapoints) or dataset_2 is None:
            dataset = dataset_1
        else:
            dataset = dataset_2
            # We want the idx to start from 0 for dataset_2
            idx = idx % int(dataset_split * total_datapoints)

        return dataset.get_item_verbose(idx), f"{dataset.split.name}_{idx}"

    for isoform_idx in tqdm(range(total_datapoints)):
        (
            (
                (
                    isoform_X_esm2_encoding_orig,
                    isoform_X_protein_len_orig,
                    _isoform_X_landmark_stains_orig,
                    _isoform_y_multilabel_orig,
                    _isoform_y_antibody_stain_orig,
                ),
                isoform_metadata,
            ),
            isoform_idx,
        ) = resolve_dataset(isoform_idx, total_datapoints)

        for cell_idx in range(total_datapoints):
            (
                (
                    (
                        _cell_X_esm2_encoding_orig,
                        _cell_X_protein_len_orig,
                        cell_X_landmark_stains_orig,
                        _cell_y_multilabel_orig,
                        cell_y_antibody_stain_orig,
                    ),
                    cell_line_metadata,
                ),
                cell_idx,
            ) = resolve_dataset(cell_idx, total_datapoints)

            if len(embedding_df.index.intersection([(isoform_idx, cell_idx)])) > 0:
                continue
            y_pred_multilabel, y_pred_antibody_stain_pred, activation = call_model(
                model,
                isoform_X_esm2_encoding_orig,
                isoform_X_protein_len_orig,
                cell_X_landmark_stains_orig,
            )
            result = pd.Series(
                {
                    "splice_variant_id": isoform_metadata["splice_isoform_id"],
                    "cell_line": cell_line_metadata["cell_line"],
                    "cell_image": cell_line_metadata["cell_image"][
                        "nuclei_channel"
                    ],
                    "activation": activation,
                    "cell_line_metadata": cell_line_metadata,
                    "isoform_metadata": isoform_metadata,
                    "y_pred_multilabel": y_pred_multilabel,
                    "y_pred_antibody_stain_pred": y_pred_antibody_stain_pred.detach().numpy(),
                    "y_antibody_stain_orig": cell_y_antibody_stain_orig,
                }
            )
            embedding_df.loc[(isoform_idx, cell_idx), :] = result
        embedding_df.to_pickle(output_pkl_file)
    embedding_df.to_pickle(output_pkl_file)
    return embedding_df
