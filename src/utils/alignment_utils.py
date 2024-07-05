import subprocess
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

# Download the muscle binary https://drive5.com/muscle/downloads_v3.htm
# Remember to use `tar -zxvf` to unpack`
MUSCLE_ALIGN_CALL = "~/muscle3.8.31_i86linux64"


def muscle_align(
    seqs,
    seq_record_name="~/_example.fasta",
    align_name="~/_align.fasta",
):
    SeqIO.write(
        [SeqRecord(Seq(seq), id=seq) for seq in seqs],
        seq_record_name,
        "fasta",
    )
    subprocess.call(
        "%s -in %s -out %s -maxiters 1"
        % (MUSCLE_ALIGN_CALL, seq_record_name, align_name),
        shell=True,
    )
    with open(align_name, "r") as f:
        raw_seqs = f.readlines()
    return [seq.strip() for seq in raw_seqs if ("#" not in seq) and (">") not in seq]