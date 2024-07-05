from torch.utils.data import DataLoader, Dataset
from pymongo import MongoClient
from torchvision import transforms
import numpy as np
from PIL import Image
import pdb
import torch
from enum import Enum
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
from src.utils.utils import get_data_path


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    EVAL = "eval"
 

CLASSES = [
    [
        "Nucleoplasm",
        "Cytosol",
        "Vesicles",
        "Plasma membrane",
        "Mitochondria",
        "Golgi apparatus",
        "Endoplasmic reticulum",
        "Nucleoli",
        "Nuclear bodies",
        "Nuclear speckles",
        "Nuclear membrane",
        "Peroxisomes",
        "Microtubules",
        "Centrosome",
        "Cytokinetic bridge",
        "Mitotic chromosome",
        "Centriolar satellite",
        "Focal adhesion sites",
        "Cell Junctions",
        "Lipid droplets",
        "Nucleoli fibrillar center",
        "Actin filaments",
        "Mitotic spindle",
        "Midbody ring",
        "Cytoplasmic bodies",
        "Nucleoli rim",
        "Midbody",
        "Intermediate filaments",
        "Aggresome",
    ]
]


class SubCellDatset(Dataset):
    def __init__(self, split, collection_name, if_alphabetical=False):
        """
            NOTE: if_alphabetical was added as a means to switch between two copies of the HPA mongo server
            The old HPA client contains collection: splice_isoform_dataset_cell_line_and_gene_split_full
                * The training set
                * Holdout 1 in the paper
            The new HPA client contains collection: random_splice_isoform_dataset 
                * Holdout 2 in the paper
        """
        self.client = MongoClient(maxPoolSize=500)
        self.split = split
        if if_alphabetical:
            self.splice_isoforms_collection = self.client.hpa_old.splice_isoforms
            self.dataset_collection = self.client.hpa_old[collection_name]
            self.len = (
                list(self.client.hpa_old[f"{collection_name}_metadata"].find({}))[0][
                    split.value + "_count"
                ]
                - 1
            )

        else:
            self.splice_isoforms_collection = self.client.hpa.splice_isoforms
            self.dataset_collection = self.client.hpa[collection_name]
            self.len = (
                list(self.client.hpa[f"{collection_name}_metadata"].find({}))[0][
                    split.value + "_count"
                ]
                - 1
            )

        self.ml_binarizer = MultiLabelBinarizer().fit(CLASSES)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        res = self.get_item_verbose(idx)
        if res is None:
            return None
        datum, _metadatum = res
        return datum

    def get_item_verbose(self, idx, filter_low_values=0.19):
        # This try catch block is because not all datapoints have esm2_representations
        # and valid landmark stains... 
        try:
            # Return ESM2 representation, subcell label, cell image
            datapoint = list(
                self.dataset_collection.find({"_id": f"{self.split.value}_{idx}"})
            )[0]
            isoform_data = list(
                self.splice_isoforms_collection.find(
                    {"_id": datapoint["splice_isoform_id"]}
                )
            )[0]

            X_esm2_encoding = pickle.loads(isoform_data["esm2_representation"]["binary"])
            X_protein_len = isoform_data["length"]
        except:
            return None
        
        X_landmark_stains = np.stack(
            (
                np.array(
                    Image.open(get_data_path(datapoint["cell_image"]["nuclei_channel"]))
                ),
                np.array(
                    Image.open(
                        get_data_path(datapoint["cell_image"]["microtubule_channel"])
                    )
                ),
                np.array(
                    Image.open(
                        get_data_path(datapoint["cell_image"]["mitochondria_channel"])
                    )
                ),
            )
        )
        if filter_low_values is not None:
            X_landmark_stains[X_landmark_stains < filter_low_values] = 0 

        # Suppress the warnings for compartment classes not in CLASSES
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_multilabel = self.ml_binarizer.transform(
                [datapoint["location_labels"].split(",")]
            )[0]
        y_antibody_stain = np.array(
            Image.open(get_data_path(datapoint["cell_image"]["antibody_channel"]))
        )
        return (
            (
                X_esm2_encoding,
                X_protein_len,
                X_landmark_stains,
                y_multilabel,
                y_antibody_stain,
            ),
            datapoint,
        )

    def __del__(self):
        print("Cleaning up...")
        self.client.close()
        print("Finished cleaning up")
