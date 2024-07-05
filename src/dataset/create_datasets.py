import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))

import random
from math import isclose
from tqdm import tqdm
from enum import Enum
from pymongo import MongoClient
import pickle
import numpy as np
from PIL import Image
import time


class SplitType(Enum):
    RANDOM = 0
    # Split along cell_lines
    CELL_LINE = 1
    # Split along gene families
    GENE_FAMILY = 2
    # Separate splice variants (i.e., every gene straddles test & train)
    SPLICE_VARIANT_FAMILY = 3
    # Split along cell_lines and genes (cell lines and gene families are split between datasets)
    CELL_LINE_AND_GENE = 4


def decide_split(roll, train_test_eval_ratios):
    if roll - train_test_eval_ratios[0] < 0:
        return "train"
    elif roll - (train_test_eval_ratios[0] + train_test_eval_ratios[1]) < 0:
        return "test"
    else:
        return "eval"


def portion_dataset(roll, train_count, test_count, eval_count, train_test_eval_ratios):
    split = decide_split(roll, train_test_eval_ratios)
    if split == "train":
        label = "train_" + str(train_count)
        train_count += 1
    elif split == "test":
        label = "test_" + str(test_count)
        test_count += 1
    else:
        label = "eval_" + str(eval_count)
        eval_count += 1
    return label, train_count, test_count, eval_count


def get_datapoint(datapoint, splice_isoforms_collection, binarizer):
    isoform_data = list(
        splice_isoforms_collection.find({"_id": datapoint["splice_isoform_id"]})
    )[0]

    X_esm2_encoding = pickle.loads(isoform_data["esm2_representation"]["binary"])
    X_protein_len = isoform_data["length"]
    X_landmark_stains = np.stack(
        (
            np.array(Image.open(datapoint["cell_image"]["nuclei_channel"])),
            np.array(Image.open(datapoint["cell_image"]["microtubule_channel"])),
            np.array(Image.open(datapoint["cell_image"]["mitochondria_channel"])),
        )
    )
    # Add transformations here to augment the dataset if necessary
    y_multilabel = binarizer.transform([datapoint["location_labels"].split(",")])[0]
    y_antibody_stain = np.array(Image.open(datapoint["cell_image"]["antibody_channel"]))
    return (
        X_esm2_encoding,
        X_protein_len,
        X_landmark_stains,
        y_multilabel,
        y_antibody_stain,
    )


def create_datasets(
    collection_name,
    split_method: SplitType,
    train_test_eval_ratios=(0.7, 0.2, 0.1),
    random_seed=0,
    overwrite=True,
    limit=5000,
    allow_missing_cell_images=True,
    save_root="cell_images",
    crop_key="single_cells_crops",
):
    """
    This method creates three new datasets: Train / Test / Eval
    Split according to split_method and saved into: collection_name appended with train / test /eval
    Additionally it creates collection_name_metadata to act as a helper to manage dataset retrieval
    """
    if allow_missing_cell_images:
        match = {
            "location_labels": {"$exists": True},
            "splice_isoform_id": {"$exists": True},
            "gene": {"$exists": True},
            "splice_isoform": {"$exists": True},
            "cell_line": {"$exists": True},
        }
    else:
        match = {
            "location_labels": {"$exists": True},
            "splice_isoform_id": {"$exists": True},
            "cell_image.nuclei_channel": {"$exists": True},
            "cell_image.microtubule_channel": {"$exists": True},
            "cell_image.mitochondria_channel": {"$exists": True},
            "cell_image.antibody_channel": {"$exists": True},
            "gene": {"$exists": True},
            "splice_isoform": {"$exists": True},
            "cell_line": {"$exists": True},
        }
    pipeline = [
        {"$limit": limit},
        {
            "$match": {
                "esm2_representation": {"$exists": True},
                "esm2_representation.binary": {"$exists": True},
                "cell_lines": {"$exists": True},
            }
        },
        {"$unwind": {"path": "$cell_lines"}},
        {"$unwind": {"path": "$cell_lines.antibodies"}},
        {"$unwind": {"path": "$cell_lines.antibodies.image_urls"}},
        {"$unwind": {"path": f"$cell_lines.antibodies.image_urls.{save_root}"}},
        {
            "$project": {
                "location_labels": "$cell_lines.antibodies.location_labels.locations",
                "splice_isoform_id": "$_id",
                # Copying the esm2 representations results for each datapoint is too big to handle...
                # So we only copy the isoform_id and look it up on the fly, essentially storing the pointer
                "cell_image": f"$cell_lines.antibodies.image_urls.{save_root}",
                "gene": "$parent gene",
                "splice_isoform": "$splice_isoform_ensemble_id",
                "cell_line": "$cell_lines.name",
            }
        },
        {
            "$match": match,
        },
    ]

    assert isclose(
        train_test_eval_ratios[0]
        + train_test_eval_ratios[1]
        + train_test_eval_ratios[2],
        1.0,
        abs_tol=1e-8,
    )
    random.seed(random_seed)

    with MongoClient() as client:
        splice_isoform_collection = client.hpa.splice_isoforms

        output_collection = client.hpa[collection_name]
        output_collection_metadata = client.hpa[collection_name + "_metadata"]

        if overwrite:
            output_collection.drop()
            output_collection_metadata.drop()

        results = splice_isoform_collection.aggregate(pipeline, allowDiskUse=True)

        train_idx, test_idx, eval_idx = 0, 0, 0
        for idx, res in tqdm(enumerate(results), total=limit):
            try:
                stats = client.hpa.command('collStats', collection_name)

                print(idx, stats['totalIndexSize'])
            except:
                print(idx)

            # Without adding some sort of control 
            # the speed of the loop crashes our pymongo server
            time.sleep(0.0001) 

            if split_method == SplitType.RANDOM:
                # We don't want the same exact sequence to be in both test & train.
                random.seed(hash(res["splice_isoform_id"]) + random_seed)
                roll = random.uniform(0.0, 1.0)
            elif split_method == SplitType.GENE_FAMILY:
                # Split train / test / eval such that gene families are divided
                random.seed(hash(res["gene"]) + random_seed)
                roll = random.uniform(0.0, 1.0)
            elif split_method == SplitType.CELL_LINE:
                # Split train / test / eval such that cell lines are divided
                random.seed(hash(res["cell_line"]) + random_seed)
                roll = random.uniform(0.0, 1.0)
            elif split_method == SplitType.CELL_LINE_AND_GENE:
                # Split train / test / eval such that cell lines & gene families are divided
                random.seed(hash(res["gene"]) + random_seed)
                gene_roll = random.uniform(0.0, 1.0)

                random.seed(hash(res["cell_line"]) + random_seed)
                cell_line_roll = random.uniform(0.0, 1.0)

                # Only keep the datapoint if both the cell line & gene rolls agree on placement
                if decide_split(gene_roll, train_test_eval_ratios) != decide_split(
                    cell_line_roll, train_test_eval_ratios
                ):
                    continue
                else:
                    roll = cell_line_roll
            elif split_method == SplitType.SPLICE_VARIANT_FAMILY:
                raise NotImplementedError(
                    "TODO: Implement Splice Variant Cousin Split!"
                )
            
            # This is the alternative to doing the ESM2 representation check in dataset.py
            # Uncommenting will incur a significant run time penalty
            # try:
            #     (
            #         X_esm2_encoding,
            #         X_protein_len,
            #         X_landmark_stains,
            #         y_multilabel,
            #         y_antibody_stain
            #     ) = get_datapoint(res, splice_isoform_collection, ml_binarizer)
            #     assert X_esm2_encoding is not None and X_protein_len is not None and X_landmark_stains is not None and y_multilabel is not None and y_antibody_stain is not None
            # except:
            #     continue

            (label, train_idx, test_idx, eval_idx) = portion_dataset(
                roll,
                train_idx,
                test_idx,
                eval_idx,
                train_test_eval_ratios,
            )
            # Feed label as the _id field for efficient lookup later
            res["_id"] = label
            output_collection.insert_one(res)
        output_collection_metadata.insert_one(
            {
                "train_count": train_idx,
                "test_count": test_idx,
                "eval_count": eval_idx,
            }
        )

create_datasets(
    "splice_isoform_cell_gene_split",
    split_method=SplitType.CELL_LINE_AND_GENE,
    train_test_eval_ratios=(0.8, 0.1, 0.1),
    limit=10000000,  # The Full dataset
    save_root="cell_images",
    crop_key="single_cells_crops",
)