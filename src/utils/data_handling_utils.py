import pathlib
import sys
import pdb

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))


import os
import numpy as np
import torch
from enum import Enum
from src.dataset.dataset import SubCellDatset, DatasetType


class Retrieval_Data(Enum):
    CELL_IMAGE = "cell_image"
    PROTEIN_SEQ = "protein_seq"
    METADATA = "metadata"
    TRUE_LABELS = "true_labels"


def initialize_datasets(collection_name, if_alphabetical=False):
    train_dataset, val_dataset, test_dataset = (
        SubCellDatset(
            split=DatasetType.TRAIN,
            collection_name=collection_name,
            if_alphabetical=if_alphabetical
        ),
        SubCellDatset(
            split=DatasetType.EVAL,
            collection_name=collection_name,
            if_alphabetical=if_alphabetical
        ),
        SubCellDatset(
            split=DatasetType.TEST,
            collection_name=collection_name,
            if_alphabetical=if_alphabetical
        ),
    )


    def get_data(datapoint_idx, retrieval_data: Retrieval_Data):
        """
        get_data() expects a datapoint_idx in the format {split}_{idx}
        e.g. "train_1" or "test_203" or "eval_12222"
        """
        if "train" in datapoint_idx:
            dataset = train_dataset
        elif "test" in datapoint_idx:
            dataset = test_dataset
        elif "eval" in datapoint_idx:
            dataset = val_dataset
        else:
            print("whats up with ", datapoint_idx)
        (
            X_esm2_encoding,
            X_protein_len,
            X_landmark_stains,
            y_multilabel,
            y_antibody_stain,
        ), metadata = dataset.get_item_verbose(int(datapoint_idx.split("_")[1]))
        if retrieval_data == Retrieval_Data.CELL_IMAGE:
            return X_landmark_stains

        if retrieval_data == Retrieval_Data.PROTEIN_SEQ:
            return (X_esm2_encoding, X_protein_len)

        if retrieval_data == Retrieval_Data.METADATA:
            return metadata

        if retrieval_data == Retrieval_Data.TRUE_LABELS:
            return y_multilabel, y_antibody_stain

    return train_dataset, val_dataset, test_dataset, get_data
