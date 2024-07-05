import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))


import tensorflow as tf
import torch
from lightning.pytorch import seed_everything

import lightning.pytorch as pl
from pymongo import MongoClient
from src.utils.data_handling_utils import Retrieval_Data
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import pdb
import contextlib
import io
from typing import List


def get_proteoform_data(
    collection_name,
    get_data,
    proteoform_id=None,
    gene_id=None,
    use_old_hpa_client=False,
):
    assert (
        proteoform_id is not None or gene_id is not None
    ), "We need either the proteoform id or the gene id specified"
    with contextlib.redirect_stdout(io.StringIO()):
        with MongoClient(maxPoolSize=500) as client:
            if use_old_hpa_client:
                dataset_collection = client.hpa_old[collection_name]
            else:
                dataset_collection = client.hpa[collection_name]

            if proteoform_id is not None:
                proteoform_cursor = dataset_collection.find(
                    {"splice_isoform_id": proteoform_id}
                )
            elif gene_id is not None:
                proteoform_cursor = dataset_collection.find({"gene": gene_id})

            for proteoform_dict in tqdm(proteoform_cursor):
                try:
                    data = get_data(
                        proteoform_dict["_id"], retrieval_data=Retrieval_Data.METADATA
                    )
                    return data
                except:
                    continue

        return None

def get_cell_lines_of_interest(
    collection_name, images_per_cell_line, get_data, seed=0, use_old_hpa_client=False
):
    with contextlib.redirect_stdout(io.StringIO()):
        seed_everything(seed)

    cell_line_grouping = None
    with MongoClient(maxPoolSize=500) as client:
        if use_old_hpa_client:
            dataset_collection = client.hpa_old[collection_name]
        else:
            dataset_collection = client.hpa[collection_name]
        cell_line_grouping = list(
            dataset_collection.aggregate(
                [{"$group": {"_id": "$cell_line", "ids": {"$push": "$_id"}}}],
                allowDiskUse=True,
            )
        )
    cell_lines_of_interest = {}

    for cell_line_dict in tqdm(cell_line_grouping):
        cell_line = cell_line_dict["_id"].replace("/", "")
        cell_images = []

        # Shuffle datapoints and accumulate the first {IMAGES_PER_CELL_LINE} num valid datapoints
        potential_cell_image_datapoints = cell_line_dict["ids"].copy()
        np.random.shuffle(potential_cell_image_datapoints)
        for cell_image_datapoint in potential_cell_image_datapoints:
            if len(cell_images) >= images_per_cell_line:
                break
            try:
                X_landmark_stains = get_data(
                    cell_image_datapoint, retrieval_data=Retrieval_Data.CELL_IMAGE
                )
            except:
                continue
            cell_images.append(cell_image_datapoint)

        cell_lines_of_interest[cell_line] = cell_images
    return cell_lines_of_interest


def get_isoforms_of_interest_by_gene_family(
    collection_name,
    total_investigated_gene_families,
    get_data,
    seed=0,
    use_old_hpa_client=False,
):
    with contextlib.redirect_stdout(io.StringIO()):
        seed_everything(seed)
    gene_grouping = None

    with MongoClient(maxPoolSize=500) as client:
        if use_old_hpa_client:
            dataset_collection = client.hpa_old[collection_name]
        else:
            dataset_collection = client.hpa[collection_name]
        gene_grouping = list(
            dataset_collection.aggregate(
                [
                    {
                        "$group": {
                            "_id": "$gene",
                            "splice_isoform_ids": {"$push": "$splice_isoform_id"},
                            "datapoint_idxs": {"$push": "$_id"},
                        }
                    }
                ],
                allowDiskUse=True,
            )
        )

    if total_investigated_gene_families is None:
        total_investigated_gene_families = len(gene_grouping)

    max_gene_family_size = 0
    isoforms_of_interest = []
    for gene_family in tqdm(
        np.random.choice(
            gene_grouping,
            size=min(total_investigated_gene_families, len(gene_grouping)),
            replace=False,
        ),
        total=min(total_investigated_gene_families, len(gene_grouping)),
    ):
        finished_isoforms = []
        # Look for the first valid datapoint in the isoform list then add it to isoforms_of_interest
        for isoform_id, datapoint_idx in zip(
            gene_family["splice_isoform_ids"], gene_family["datapoint_idxs"]
        ):
            if isoform_id in finished_isoforms:
                continue
            try:
                get_data(datapoint_idx, retrieval_data=Retrieval_Data.PROTEIN_SEQ)
            except:
                continue
            isoforms_of_interest.append(
                f"{isoform_id}.{gene_family['_id']} {datapoint_idx}"
            )
            finished_isoforms.append(isoform_id)
        if len(finished_isoforms) > max_gene_family_size:
            max_gene_family_size = len(finished_isoforms)
            print(f"new max gene family size is {max_gene_family_size}")

    return isoforms_of_interest


def get_isoforms_of_interest(
    collection_name,
    total_investigated_isoforms,
    get_data,
    seed=0,
    use_old_hpa_client=False,
):
    with contextlib.redirect_stdout(io.StringIO()):
        seed_everything(seed)
    isoform_grouping = None

    with MongoClient(maxPoolSize=500) as client:
        if use_old_hpa_client:
            dataset_collection = client.hpa_old[collection_name]
        else:
            dataset_collection = client.hpa[collection_name]
        isoform_grouping = list(
            dataset_collection.aggregate(
                [{"$group": {"_id": "$splice_isoform_id", "ids": {"$push": "$_id"}}}],
                allowDiskUse=True,
            )
        )
    if total_investigated_isoforms is None:
        total_investigated_isoforms = len(isoform_grouping)

    isoforms_of_interest = []
    for isoform in tqdm(
        np.random.choice(
            isoform_grouping,
            size=min(total_investigated_isoforms, len(isoform_grouping)),
            replace=False,
        ),
        total=min(total_investigated_isoforms, len(isoform_grouping)),
    ):
        # Look for the first valid datapoint in the isoform list then add it to isoforms_of_interest
        for isoform_datapoint_idx in isoform["ids"]:
            try:
                get_data(
                    isoform_datapoint_idx, retrieval_data=Retrieval_Data.PROTEIN_SEQ
                )
            except:
                continue
            isoforms_of_interest.append(f"{isoform['_id']} {isoform_datapoint_idx}")
            break
    return isoforms_of_interest


def construct_embedding_df(
    output_pkl_file, isoforms_of_interest, cell_lines_of_interest
):
    assert not (os.path.exists(output_pkl_file)) or os.path.isdir(
        output_pkl_file
    ), "The output file is not empty!"
    embedding_df = pd.DataFrame(
        columns=[
            "cell_line",
            "cell_image_datapoints",
        ]
        + isoforms_of_interest
    )
    embedding_df.set_index(["cell_line", "cell_image_datapoints"], inplace=True)

    for cell_line, datapoints in cell_lines_of_interest.items():
        embedding_df.loc[(cell_line, ",".join(datapoints)), :] = np.NaN

    return embedding_df


def get_model_latent_representation(
    loaded_model, X_esm2_encoding, X_protein_len, X_landmark_stains, device
):
    activation = {}

    def get_activation(name, rename):
        def hook(model, input, output):
            activation[rename] = output.detach()

        return hook

    loaded_model.cell_inpainting_unet.upsample_layer2.register_forward_hook(
        get_activation("0", "joint_embedding_after_upsample_layer2")
    )
    loaded_model.to(device)
    X_esm2_encoding = X_esm2_encoding.to(device)
    X_protein_len = X_protein_len.to(device)
    X_landmark_stains = X_landmark_stains.to(device)
    (
        _y_pred_antibody_stain,
        _y_pred_multilabel,
        _y_pred_ranked,
    ) = loaded_model.predict_step(
        (
            X_esm2_encoding,
            X_protein_len,
            X_landmark_stains,
            None,
            None,
        ),
        batch_idx=0,
    )
    return activation["joint_embedding_after_upsample_layer2"]


def replace_with_boolean(obj):
    return isinstance(
        obj,
        (np.ndarray, bool, torch.Tensor),
    )


def stitch_dataframe_from_folder(folder_path, bool_values=True):
    # Rather than loading each of the cell values (which can be ~0.5M floats)
    # if bool_values = True, then just fill in the value with a boolean for book-keeping...
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pkl"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_pickle(file_path)
            if bool_values:
                df = df.applymap(replace_with_boolean)
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)



def batch_call(
    loaded_model,
    embedding_df,
    output_pkl_file,
    get_data,
    pca=None,
    pseudo_pca=None,
    save_frequency=20,
    device=torch.device("cuda:0"),
):
    """
    There are two modes for calling batch_call, either output_pkl_file is a FILE in which case we just save a pandas pkl file
    or output_pkl is a FOLDER, in which case we save every isoform as its own individual file in the folder.

    Then if pca is provided, we also will take the PCA of the embedding before saving (to save space)
    """
    loaded_model.to(device)

    save_dir = output_pkl_file if os.path.isdir(output_pkl_file) else None
    if pseudo_pca is not None:
        pseudo_pca.eval()
        pseudo_pca.encoder.eval()
        pseudo_pca.to(device)

    for idx, (_, row) in tqdm(
        enumerate(embedding_df.T.iterrows()), total=len(embedding_df.T)
    ):
        isoform = row.name
        isoform_datapoint_idx = isoform.split(" ")[1]
        save_dir_filename = (
            os.path.join(save_dir, f"{isoform}.pkl") if save_dir is not None else None
        )

        for cell_line, cell_image_datapoint_idxs in list(row.index):
            print("we are looking at cell_line ", cell_line)
            if isinstance(
                embedding_df.loc[(cell_line, cell_image_datapoint_idxs)][isoform],
                (np.ndarray, torch.Tensor, bool, List),
            ) or (save_dir_filename is not None and os.path.exists(save_dir_filename)):
                continue
            # Get the encodings
            X_landmark_stains = torch.from_numpy(
                np.stack(
                    [
                        get_data(
                            cell_image_datapoint_idx,
                            retrieval_data=Retrieval_Data.CELL_IMAGE,
                        )
                        for cell_image_datapoint_idx in cell_image_datapoint_idxs.split(
                            ","
                        )
                    ]
                )
            )
            X_esm2_encoding, X_protein_len = get_data(
                isoform_datapoint_idx, retrieval_data=Retrieval_Data.PROTEIN_SEQ
            )
            X_esm2_encodings = torch.from_numpy(
                np.repeat(
                    tf.expand_dims(X_esm2_encoding, axis=0),
                    X_landmark_stains.shape[0],
                    axis=0,
                )
            )
            X_protein_lens = torch.from_numpy(
                np.repeat(
                    np.array([X_protein_len]).reshape((1)),
                    X_landmark_stains.shape[0],
                    axis=0,
                )
            )
            # Calculate embeddings
            averaged_embedding = torch.mean(
                get_model_latent_representation(
                    loaded_model,
                    X_esm2_encodings,
                    X_protein_lens,
                    X_landmark_stains,
                    device,
                ).squeeze(),
                axis=0,
            )
            # Get PCA embeddings (if specified)
            if pseudo_pca is not None:
                flattened_embedding = averaged_embedding.flatten()
                averaged_embedding = pseudo_pca.encoder(flattened_embedding).tolist()
            elif pca is not None:
                flattened_embedding = np.array(
                    [
                        tensor.flatten()
                        for tensor in averaged_embedding.numpy().flatten()
                    ]
                )
                averaged_embedding = pca.transform(flattened_embedding.reshape(1, -1))
            embedding_df.loc[(cell_line, cell_image_datapoint_idxs)][
                isoform
            ] = averaged_embedding

        if save_dir is not None:
            isoform_df = embedding_df[isoform].to_frame()
            isoform_df.to_pickle(save_dir_filename)
            # Replace with booleans for sake of less memory use
            embedding_df[isoform] = embedding_df[isoform].apply(replace_with_boolean)
        elif idx % save_frequency == 0:
            embedding_df.to_pickle(output_pkl_file)

        print(
            "how much of the total dataframe memory limit (100 GB) are we using up? ",
            embedding_df.memory_usage(deep=True).sum() / (100 * 1e9),
        )
    return embedding_df
