"""
Within this is our full pipeline for scraping the Human Protein Atlas, 
joining the data with Ensembl sequences, getting ESM-2 representations,
image preprocessing / cropping, and formatting all the data into a MongoDB library.

To run:
1. Setup your mongo server & create a database named "hpa" (tutorials at https://www.mongodb.com/)
2. Add the path where you'd like to save the images & mongo servers to utils.get_data_path()
3. Download the tabular data from https://www.proteinatlas.org/download/subcellular_location.tsv.zip
   (the scripts use the list of genes from the tabular data as an index)
4. Specify the path to the tabular data within build_datasets() as the argument (subcell_loc_file) 
5. Run "python download_data.py" from command line.

It may take multiple days to run, but afterwards it should generate two new collections in your mongo database:
    * hpa.genes - which is just the uploaded subcellular tabular dataset from HPA
    * hpa.splice_isoforms - which will have all of the per proteoform data organized hierarchically:
If you need any help or have any questions reach out to yitongt@mit.edu
"""

import numpy as np
import pathlib
import sys
import re

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))

import re
import multiprocessing
import random
import os
import pickle
from pymongo import MongoClient
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from io import BytesIO

import itertools
import requests
import ensembl_rest
from src.utils.utils import get_data_path
from IPython.utils import io
from src.utils.esm2_utils import get_esm2_representation
from PIL import Image, ImageOps
import numpy as np
import os
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.exposure import rescale_intensity
from scipy.ndimage.measurements import center_of_mass
from functools import partial
import cv2
from scipy.linalg import lstsq

DATASET_FOLDER = "."

def get_splice_variant_subset(
    splice_isoform_ids,
    num_isoform_images,
    shoe_in_genes=[], # Specify genes you definitely want in the subset
):
    """
    Getting a random subset of genes like this rather than using "$sample"

    since $sample runs into a whole host of memory & performance issues,
    b/c it first must sort all of the isoforms it is not a viable method of getting 
    a random subset of splice variants.

    cursor = collection.aggregate(
        [
            {"$match": {"_id": {"$in": splice_isoform_ids}}},
            {"$sample": {"size": num_isoform_images}},
        ],
        allowDiskUse=True,
        batchSize=1,
    )
    """

    def shoe_in_check(splice_isoform_id):
        return any(
            [shoe_in_prefix in splice_isoform_id for shoe_in_prefix in shoe_in_genes]
        )

    random.seed(42)
    shoe_in_splice_isoform_ids = [id for id in splice_isoform_ids if shoe_in_check(id)]
    splice_isoform_ids_subset = (
        random.sample(splice_isoform_ids, num_isoform_images)
        + shoe_in_splice_isoform_ids
    )
    return splice_isoform_ids_subset


def stem_hpa_image_url(image_url):
    # Take the stem of the image_url, meaning
    # remove the default 'blue_red_green.jpg' coloring
    stem_idx = image_url.find("blue_red_green.jpg")
    if stem_idx < 0:
        print("error in ", image_url)
    return image_url[:stem_idx]


def construct_lightfield_image_foldername(image_directory, isoform):
    return f"{image_directory}/{isoform['parent gene name']}-{isoform['parent gene']}"


def construct_singlecell_image_foldername(
    image_directory, isoform, folder="single_cells_crops"
):
    return os.path.join(
        construct_lightfield_image_foldername(image_directory, isoform),
        folder,
    )


def get_stain_signatures(image):
    hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

    # Red hues wrap around in HSV format therefore need two masks
    red_lower_1 = np.array([170, 30, 30])
    red_upper_1 = np.array([179, 150, 150])
    red_lower_2 = np.array([0, 30, 30])
    red_upper_2 = np.array([10, 150, 150])

    blue_lower = np.array([110, 30, 30])
    blue_upper = np.array([130, 150, 150])

    green_lower = np.array([50, 30, 30])
    green_upper = np.array([70, 150, 150])

    yellow_lower = np.array([20, 30, 30])
    yellow_upper = np.array([40, 150, 150])

    red_mask = cv2.inRange(hsv_image, red_lower_1, red_upper_1) | cv2.inRange(
        hsv_image, red_lower_2, red_upper_2
    )
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    # Extract pixels within the color ranges
    red_pixels = hsv_image[np.where(red_mask != 0)]
    blue_pixels = hsv_image[np.where(blue_mask != 0)]
    green_pixels = hsv_image[np.where(green_mask != 0)]
    yellow_pixels = hsv_image[np.where(yellow_mask != 0)]

    # Before taking the average, first convert to RGB (so the red wrap around is not an issue)
    red_signature = (
        cv2.cvtColor(np.uint8([red_pixels]), cv2.COLOR_HSV2RGB).mean(axis=1)[0] / 255
    )
    blue_signature = (
        cv2.cvtColor(
            np.uint8([blue_pixels]),
            cv2.COLOR_HSV2RGB,
        ).mean(
            axis=1
        )[0]
        / 255
    )
    green_signature = (
        cv2.cvtColor(
            np.uint8([green_pixels]),
            cv2.COLOR_HSV2RGB,
        ).mean(
            axis=1
        )[0]
        / 255
    )
    yellow_signature = (
        cv2.cvtColor(
            np.uint8([yellow_pixels]),
            cv2.COLOR_HSV2RGB,
        ).mean(
            axis=1
        )[0]
        / 255
    )

    # Normalize the signatures
    red_signature = red_signature * (1 / red_signature.max())
    blue_signature = blue_signature * (1 / blue_signature.max())
    green_signature = green_signature * (1 / green_signature.max())
    yellow_signature = yellow_signature * (1 / yellow_signature.max())

    return np.array(
        [
            red_signature,
            green_signature,
            blue_signature,
            yellow_signature,
        ]
    )


def unmix_spectra(
    collection,
    splice_isoform_ids,
    image_directory,
    refresh_session,
    num_isoform_images,
    overwrite=False,
):
    """
    This method performs linear spectral unmixing on the landmark (red, blue, and yellow) stains.
    Critically this is to ensure the target antibody stain does not leak through to the training labels.
    """

    def process_filename(filename):
        filepath = filename.split("images/")[-1]
        return os.path.join(image_directory, filepath)

    def flatten_image(image):
        return image.reshape((image.shape[0] * image.shape[1], image.shape[2])).T

    def unflatten_image(flattened_image):
        orig_img_dim = int(np.sqrt(flattened_image.shape[1]))
        return flattened_image.reshape(
            (
                flattened_image.shape[0],
                orig_img_dim,
                orig_img_dim,
            )
        )

    def process_individual_stain(unmixed_image_stain):
        return (
            rescale_intensity(
                unmixed_image_stain.astype(np.float32),
                out_range=(0, 1),
            )
            * 255
        ).astype(np.uint8)

    def unmix(
        image, debug_output_folder=DATASET_FOLDER, debug=False
    ):
        # Calculate spectral signatures for each
        spectral_signatures = get_stain_signatures(image)

        # Unmix the image!
        fractions, _, _, _ = lstsq(
            spectral_signatures.T,
            flatten_image(image),
        )
        # Clamp the unmixed array before casting to uint to avoid overflow
        # This is very important!
        fractions[fractions > 1] = 1
        fractions[fractions < 0] = 0

        unmixed_image = unflatten_image(fractions)

        result = (
            process_individual_stain(unmixed_image[0]),  # red
            process_individual_stain(unmixed_image[1]),  # green
            process_individual_stain(unmixed_image[2]),  # blue
            process_individual_stain(unmixed_image[3]),  # yellow
        )
        if not (debug):
            return result
        else:
            # original images
            Image.fromarray((image[:, :, 0] * 255).astype(np.uint8)).convert(
                "RGB"
            ).save(f"{debug_output_folder}/orig_red.jpg")
            Image.fromarray((image[:, :, 1] * 255).astype(np.uint8)).convert(
                "RGB"
            ).save(f"{debug_output_folder}/orig_green.jpg")
            Image.fromarray((image[:, :, 2] * 255).astype(np.uint8)).convert(
                "RGB"
            ).save(f"{debug_output_folder}/orig_blue.jpg")
            # unmixed images
            Image.fromarray(result[0]).convert("RGB").save(
                f"{debug_output_folder}/unmixed_red.jpg"
            )
            Image.fromarray(result[1]).convert("RGB").save(
                f"{debug_output_folder}/unmixed_green.jpg"
            )
            Image.fromarray(result[2]).convert("RGB").save(
                f"{debug_output_folder}/unmixed_blue.jpg"
            )
            Image.fromarray(result[3]).convert("RGB").save(
                f"{debug_output_folder}/unmixed_yellow.jpg"
            )

            return result

    splice_isoform_ids_subset = get_splice_variant_subset(
        splice_isoform_ids, num_isoform_images
    )
    cursor = collection.find(
        {"_id": {"$in": splice_isoform_ids_subset}}, no_cursor_timeout=True
    )
    for isoform in tqdm(
        cursor,
        total=len(splice_isoform_ids_subset),
    ):
        save_folder = construct_lightfield_image_foldername(image_directory, isoform)
        refresh_session()
        for cell_line in isoform.get("cell_lines", []):
            for antibody in cell_line.get("antibodies", []):
                for image_data in antibody.get("image_urls", []):
                    if (
                        not ("full_lightfield" in image_data)
                        or len(image_data["full_lightfield"]) == 0
                    ):
                        # Skip missing lightfield
                        continue

                    if (
                        not (overwrite)
                        and len(image_data.get("unmixed_lightfield", {})) > 0
                    ):
                        # Avoid overwrite
                        continue

                    def process_stain(
                        collection, color, stain, filename, save_image=True
                    ):
                        if save_image:
                            Image.fromarray(stain).convert("RGB").save(filename)
                        collection.update_one(
                            filter={"_id": isoform["_id"]},
                            update={
                                "$set": {
                                    "cell_lines.$[elem1].antibodies.$[elem2].image_urls.$[elem3].unmixed_lightfield."
                                    + color: filename
                                }
                            },
                            array_filters=[
                                {"elem1.name": cell_line["name"]},
                                {"elem2.antibody_id": antibody["antibody_id"]},
                                {"elem3.image_url": image_data["image_url"]},
                            ],
                        )

                    try:
                        _, filename = os.path.split(
                            image_data["full_lightfield"]["blue_red_green"]
                        )
                        rgb_image = np.array(
                            Image.open(
                                process_filename(
                                    image_data["full_lightfield"]["blue_red_green"]
                                )
                            )
                        )
                        rgb_image = rgb_image / 255
                        red_stain, green_stain, blue_stain, yellow_stain = unmix(
                            rgb_image
                        )
                    except:
                        continue

                    process_stain(
                        collection,
                        "red",
                        red_stain,
                        f"{save_folder}/{filename.replace('blue_red_green', 'unmixed_red')}",
                    )
                    process_stain(
                        collection,
                        "blue",
                        blue_stain,
                        f"{save_folder}/{filename.replace('blue_red_green', 'unmixed_blue')}",
                    )
                    process_stain(
                        collection,
                        "green",
                        green_stain,
                        f"{save_folder}/{filename.replace('blue_red_green', 'unmixed_green')}",
                    )
                    """
                        NOTE: Unmixing the yellow stain from the YELLOW readout is near impossible
                        it lacks the appropriate levels of Red / Green / Blue stains to balance the
                        linear unmixing. And adding it to the RGB stains ends up with hugely muddied
                        unmixed strains since the yellow channel is a mixture of green & blue.

                        And using the unmixed "yellow" strain from the RGB channel just essentially 
                        gives us the red channel back. We instead pass through the original unmixed yellow 
                        channel and let the low pass filter later in the pipeline handle any bleed through.
                    """
                    if os.path.exists(
                        f"{save_folder}/{filename.replace('blue_red_green', 'unmixed_yellow')}"
                    ):
                        os.remove(
                            f"{save_folder}/{filename.replace('blue_red_green', 'unmixed_yellow')}"
                        )
                        process_stain(
                            collection, "yellow", None, "DELETED_FILE", save_image=False
                        )


def download_images(
    collection,
    splice_isoform_ids,
    image_directory,
    refresh_session,
    num_isoform_images,
    colors=["yellow", "blue_red_green"],
    overwrite=False,
):
    """
    This method downloads all of the images for a subset of splice variants
    """
    if not os.path.exists(image_directory):
        try:
            os.mkdir(image_directory)
            print("we are running into an issue with ", image_directory)
        except:
            pass

    splice_isoform_ids_subset = get_splice_variant_subset(
        splice_isoform_ids, num_isoform_images
    )
    cursor = collection.find(
        {"_id": {"$in": splice_isoform_ids_subset}}, no_cursor_timeout=True
    )

    for isoform in tqdm(
        cursor,
        total=len(splice_isoform_ids_subset),
    ):
        refresh_session()
        save_folder = construct_lightfield_image_foldername(image_directory, isoform)
        if not os.path.exists(save_folder):
            try:
                os.mkdir(save_folder)
            except:
                pass

        for cell_line in isoform.get("cell_lines", []):
            for antibody in cell_line.get("antibodies", []):
                for image_data in antibody.get("image_urls", []):
                    for color in colors:
                        stemmed_url = stem_hpa_image_url(image_data["image_url"])
                        image_url = f"{stemmed_url}{color}.jpg"
                        imagename = image_url.rsplit("/", 1)[-1]
                        imagename = f"{cell_line['name']}_{imagename}".replace("/", "")

                        # Save the image to disk
                        filename = f"{save_folder}/{imagename}"
                        if (os.path.exists(filename) and overwrite) or (
                            not os.path.exists(filename)
                        ):
                            try:
                                response = requests.get(image_url)
                                # Load the image from the response content
                                image = Image.open(BytesIO(response.content))
                                with open(filename, "wb") as f:
                                    image.save(f, format="JPEG")
                            except:
                                # if one color fails, don't try the others
                                print(
                                    "danger, failed to save lightfield image ", filename
                                )
                                continue

                        collection.update_one(
                            filter={"_id": isoform["_id"]},
                            update={
                                "$set": {
                                    "cell_lines.$[elem1].antibodies.$[elem2].image_urls.$[elem3].full_lightfield."
                                    + color: filename
                                }
                            },
                            array_filters=[
                                {"elem1.name": cell_line["name"]},
                                {"elem2.antibody_id": antibody["antibody_id"]},
                                {"elem3.image_url": image_data["image_url"]},
                            ],
                        )
    cursor.close()


def antibody_scrape(document, hpa_version):
    """
    This method scrapes the antibody information for a given gene
    """
    gene = document.get("Gene")
    gene_name = document.get("Gene name")
    url = f"https://{hpa_version}.proteinatlas.org/{gene}-{gene_name}/antibody"
    html_data = requests.get(url).content
    soup = BeautifulSoup(html_data, "lxml")

    if soup.find("span", string="Matching transcripts") is None:
        # skip genes without splice variants
        return None

    antibody_data = soup.find("span", string="Product name").parent.parent.find_all(
        "td"
    )
    splice_variant_data = soup.find(
        "span", string="Matching transcripts"
    ).parent.parent.find_all("td")
    splice_variants_by_antibody_df = pd.DataFrame(
        list(
            itertools.chain(
                *[
                    [
                        {
                            "antibody_name": antibody.text,
                            "splice_isoform_ensemble_id": re.search(
                                r"(ENSP\d+)", splice_variant.text
                            )
                            .group(0)
                            .strip(),
                            "splice_isoform_hpa_id": splice_variant.text[
                                : splice_variant.text.index(" ")
                            ],
                            "version": hpa_version,
                            "splice_isoform_url": splice_variant["href"],
                        }
                        for splice_variant in splice_variants.find_all("a")
                    ]
                    for antibody, splice_variants in zip(
                        antibody_data, splice_variant_data
                    )
                ]
            )
        )
    )

    def aggregation_func(group):
        return pd.Series(
            {
                "matching_antibodies": tuple(group["antibody_name"]),
            }
        )

    return splice_variants_by_antibody_df.groupby(
        [
            "splice_isoform_ensemble_id",
            "splice_isoform_url",
            "splice_isoform_hpa_id",
            "version",
        ],
        as_index=False,
    ).apply(aggregation_func)


def scrape_image_urls(hpa_gene_id, version, color):
    # Example xml and image url:
    # https://www.proteinatlas.org/ENSG00000147421.xml
    # https://images.proteinatlas.org/58586/1060_C8_6_blue_red_green.jpg
    url = f"https://{version}.proteinatlas.org/{hpa_gene_id}.xml"
    xml_data = requests.get(url).content
    soup = BeautifulSoup(xml_data, "xml")
    scraped_subcell_images = []
    for imageUrl in soup.findChildren("imageUrl", recursive=True):
        if (
            imageUrl.parent.name == "image"
            and imageUrl.parent.parent.name == "assayImage"
            and imageUrl.parent.parent.parent.name == "data"
            and imageUrl.parent.parent.parent.parent.name == "subAssay"
        ):
            image_url = f"{stem_hpa_image_url(imageUrl.text)}{color}.jpg"
            cell_line = imageUrl.parent.parent.parent.cellLine.text
            antibody_id = imageUrl.parent.parent.parent.parent.parent.parent["id"]
            antigen_seq = (
                imageUrl.parent.parent.parent.parent.parent.parent.antigenSequence.text
            )
            if version in ("v19", "v20", "v21", "v22"):
                organ = imageUrl.parent.parent.parent.cellLine["organ"]
                cellosaurus_id = imageUrl.parent.parent.parent.cellLine["cellosaurusID"]
            else:
                organ, cellosaurus_id = None, None
            scraped_subcell_images.append(
                {
                    "image_id": image_url.rsplit("/", 1)[-1],
                    "image_url": image_url,
                    "cell_line": cell_line.replace("-", "").replace(" ", ""),
                    "organ": organ,
                    "cellosaurusID": cellosaurus_id,
                    "antibody_hpa_id": antibody_id,
                    "antigen_sequence": antigen_seq,
                    "version": version,
                }
            )
    return scraped_subcell_images


def update_cellline_antibody_images(splice_isoform_collection, document, image_df):
    """
    This method is a bit complicated but what it's doing is:
    * Find splice isoforms that match the antibody
    * Add a cell line if the cell line has not been added yet
    * Add an antibody for said cell line
      if the antibody has not been added yet for cell_line
    * Add an image_url for said antibody for said cell line
      if the image_url has not been added yet for said antibody for said cell_line
    We end up with a huge block of code since PyMongo can be a bit clunky
    """
    # Drop all rows in image_df without a matching antigen_sequence
    image_df = image_df[image_df["antigen_sequence"] != ""]

    for antibody_id in set(image_df["antibody_hpa_id"]):
        antigen_sequence = set(
            image_df[image_df["antibody_hpa_id"] == antibody_id]["antigen_sequence"]
        ).pop()

        filter = {
            "parent gene": document.get("Gene"),
            "matching_antibodies": {"$in": [antibody_id]},
        }
        # Empty our cell lines so we don't run into issues of adding the same element twice
        splice_isoform_collection.update_many(
            filter=filter,
            update={"$pull": {"cell_lines": {}}},
        )

        subset_image_df = image_df.loc[image_df["antibody_hpa_id"] == antibody_id]
        for cell_line in set(subset_image_df["cell_line"]):
            organ = set(image_df[image_df["cell_line"] == cell_line]["organ"]).pop()
            cellosaurus_id = set(
                image_df[image_df["cell_line"] == cell_line]["cellosaurusID"]
            ).pop()
            splice_isoform_collection.update_many(
                filter=filter,
                update={
                    "$addToSet": {
                        "cell_lines": {
                            "name": cell_line,
                            "organ": organ,
                            "cellosaurusID": cellosaurus_id,
                            "antibodies": [],
                        }
                    }
                },
            )

    for antibody_id in set(image_df["antibody_hpa_id"]):
        antigen_sequence = set(
            image_df[image_df["antibody_hpa_id"] == antibody_id]["antigen_sequence"]
        ).pop()

        filter = {
            "parent gene": document.get("Gene"),
            "matching_antibodies": {"$in": [antibody_id]},
        }
        subset_image_df = image_df.loc[image_df["antibody_hpa_id"] == antibody_id]
        for cell_line in set(subset_image_df["cell_line"]):
            organ = set(image_df[image_df["cell_line"] == cell_line]["organ"]).pop()
            cellosaurus_id = set(
                image_df[image_df["cell_line"] == cell_line]["cellosaurusID"]
            ).pop()

            splice_isoform_collection.update_many(
                filter=filter,
                update={
                    "$addToSet": {
                        "cell_lines.$[elem].antibodies": {
                            "antibody_id": antibody_id,
                            "antigen_sequence": antigen_sequence,
                        }
                    }
                },
                array_filters=[
                    {
                        "elem.name": cell_line,
                        "elem.organ": organ,
                        "elem.cellosaurusID": cellosaurus_id,
                    }
                ],
            )

            for image_url in set(
                subset_image_df[subset_image_df["cell_line"] == cell_line]["image_url"]
            ):
                all_versions = set(
                    subset_image_df[subset_image_df["image_url"] == image_url][
                        "all_versions"
                    ]
                ).pop()
                splice_isoform_collection.update_many(
                    filter=filter,
                    update={
                        "$addToSet": {
                            "cell_lines.$[elem1].antibodies.$[elem2].image_urls": {
                                "image_url": image_url,
                                "versions": all_versions,
                            }
                        }
                    },
                    array_filters=[
                        {
                            "elem1.name": cell_line,
                            "elem1.organ": organ,
                            "elem1.cellosaurusID": cellosaurus_id,
                        },
                        {
                            "elem2.antibody_id": antibody_id,
                            "elem2.antigen_sequence": antigen_sequence,
                        },
                    ],
                )


def download_cell_lines_antibodies_image_urls(
    gene_collection,
    splice_isoform_collection,
    gene_ids,
    hpa_versions=["v22"],
    overwrite=False,
    color="blue_red_green",
):
    """
    This method downloads all antibody image urls for each gene in gene_ids
    """
    def image_aggregation_func(group):
        return pd.Series(
            {
                "all_versions": tuple(group["version"]),
                # Get the most recent image url, organ, cellosaurusID, and antigen_sequence
                "image_url": list(group["image_url"])[-1],
                "organ": list(group["organ"])[-1],
                "cellosaurusID": list(group["cellosaurusID"])[-1],
                "antigen_sequence": list(group["antigen_sequence"])[-1],
            }
        )

    cursor = gene_collection.find({"_id": {"$in": gene_ids}}, no_cursor_timeout=True)
    for document in tqdm(
        cursor,
        total=gene_collection.count_documents({"_id": {"$in": gene_ids}}),
    ):
        if not overwrite and (
            splice_isoform_collection.count_documents(
                {
                    "parent gene": document.get("Gene"),
                    "cell_lines": {"$size": 0},
                }
            )
            == 0
        ):
            continue

        output = []
        for version in hpa_versions:
            output.append(
                pd.DataFrame(scrape_image_urls(document["Gene"], version, color))
            )

        try:
            image_df = (
                pd.concat(output)
                .groupby(
                    [
                        "image_id",
                        "antibody_hpa_id",
                        "cell_line",
                    ],
                    as_index=False,
                )
                .apply(image_aggregation_func)
            )
        except:
            continue
        update_cellline_antibody_images(splice_isoform_collection, document, image_df)
    cursor.close()


def populate_genes(filename, collection):
    """
    Inserts the HPA tabular subcellular data into MongoDB (https://www.proteinatlas.org/download/subcellular_location.tsv.zip)
    This is necessary as a starting point for the scraping script to know which genes are available
    """
    try:
        subcell_loc_df = pd.read_csv(filename, delimiter="\t")
        subcell_loc_df["_id"] = subcell_loc_df["Gene"]
        collection.insert_many(subcell_loc_df.to_dict("records"))
    except:
        pass


def download_seqs_from_ensemble(collection, splice_isoform_ids):
    """
    This method downloads the sequence information from Ensembl for all proteoforms in the collection
    """
    cursor = collection.find(
        {"_id": {"$in": splice_isoform_ids}}, no_cursor_timeout=True
    )
    for isoform in tqdm(
        cursor,
        total=collection.count_documents({"_id": {"$in": splice_isoform_ids}}),
    ):
        if isoform.get("sequence") is None or isoform.get("sequence") == "":
            try:
                seq = ensembl_rest.sequence_id(
                    isoform["splice_isoform_ensemble_id"].strip(),
                    headers={"content-type": "text/plain"},
                )
            except:
                # Sequence did not load... gonna have to update with another option.
                seq = ""
            collection.update_one(
                {"_id": isoform.get("_id")},
                {"$set": {"sequence": seq}},
            )
    cursor.close()


def calculate_esm2_embeddings(collection, splice_isoform_ids):
    """
    This method computes the ESM-2 embedding for all the proteoforms in the collection
    Only run after already scraping the sequences with Ensembl by running download_seqs_from_ensemble()
    """
    cursor = collection.find(
        {"_id": {"$in": splice_isoform_ids}}, no_cursor_timeout=True
    )
    for isoform in tqdm(
        cursor,
        total=collection.count_documents({"_id": {"$in": splice_isoform_ids}}),
    ):
        if (
            isoform.get("esm2_representation") is None
            or isoform.get("esm2_representation")["binary"] is None
        ) and isoform.get("sequence"):
            try:
                with io.capture_output() as _captured:
                    X, x_len = get_esm2_representation(
                        isoform.get("_id"), isoform.get("sequence")
                    )
                    # X.element_size() * X.nelement() == 10240000
                    # getsizeof(pickle.dumps(X, protocol=2)) == 10867981
                    # So this is a reasonable way to store the data space-wise
                    esm2_representation = pickle.dumps(X, protocol=2)
                    length = x_len.cpu().detach().item()
            except Exception as e:
                print("oops no esm2 ", e)
                length = None
                esm2_representation = None
            collection.update_one(
                {"_id": isoform.get("_id")},
                {
                    "$set": {
                        "esm2_representation": {
                            "binary": esm2_representation,
                        },
                        "length": length,
                    }
                },
            )
    cursor.close()


def scrape_subcell_labels(document, hpa_version):
    """
    This method scrapes the subcellular location label(s) per antibody / cell line / version
    e.g., https://www.proteinatlas.org/ENSG00000000003-TSPAN6/subcellular
    """
    gene = document.get("Gene")
    gene_name = document.get("Gene name")
    url = f"https://{hpa_version}.proteinatlas.org/{gene}-{gene_name}/subcellular"
    html_data = requests.get(url).content
    soup = BeautifulSoup(html_data, "lxml")
    scraped_labels = []

    if hpa_version in ("v16", "v17", "v18"):
        location_idx = 3
    elif hpa_version in ("v19", "v20", "v21", "v22"):
        location_idx = 4

    try:
        trs = soup.find("td", {"class": "cellThumbs"}).parent.parent.find_all("tr")
        for idx, tr in enumerate(trs):
            cell_thumb = tr.find("td", {"class": "cellThumbs"})

            if cell_thumb is not None:
                tds = cell_thumb.parent.find_all("td")
                cell_line = tds[2].text.replace("-", "").replace(" ", "")
                antibody = re.sub("\s+", "", tds[1].text)
                locations = [tds[location_idx].text]
                for relevant_idx in range(1, int(tds[0]["rowspan"])):
                    locations.append(trs[idx + relevant_idx].td.text)
                locations.sort()  # Sort so the list avoids collisions when grouping
                scraped_labels.append(
                    {
                        "locations": ",".join(locations),
                        "antibody_id": antibody,
                        "cell_line": cell_line,
                        "version": hpa_version,
                    }
                )
    except:
        print("issue with ", url)
    return scraped_labels


def download_per_cell_line_subcell_labels(
    gene_collection, splice_isoform_collection, gene_ids, hpa_versions=["v22"]
):
    """
    This method scrapes the subcellular localization labels for every cell line & antibody
    combination across hpa_versions for a certain gene_collection
    """
    def label_aggregation_func(group):
        return pd.Series(
            {
                "all_versions": tuple(group["version"]),
                # Get the most recent locations by version
                "locations": list(group["locations"])[-1],
            }
        )

    cursor = gene_collection.find({"_id": {"$in": gene_ids}}, no_cursor_timeout=True)
    for document in tqdm(
        cursor,
        total=gene_collection.count_documents({"_id": {"$in": gene_ids}}),
    ):
        output = []
        for version in hpa_versions:
            output.append(pd.DataFrame(scrape_subcell_labels(document, version)))
        try:
            image_df = (
                pd.concat(output)
                .groupby(
                    [
                        "antibody_id",
                        "cell_line",
                    ],
                    as_index=False,
                )
                .apply(label_aggregation_func)
            )
            for idx, row in image_df.iterrows():
                splice_isoform_collection.update_many(
                    filter={
                        "parent gene": document.get("Gene"),
                        "matching_antibodies": {"$in": [row["antibody_id"]]},
                    },
                    update={
                        "$set": {
                            "cell_lines.$[elem1].antibodies.$[elem2].location_labels": {
                                "locations": row["locations"],
                                "versions": row["all_versions"],
                            }
                        }
                    },
                    array_filters=[
                        {
                            "elem1.name": row["cell_line"],
                        },
                        {
                            "elem2.antibody_id": row["antibody_id"],
                        },
                    ],
                )
        except:
            continue

    cursor.close()


def process_patch(stain, c1, c2, c3, c4):
    return rescale_intensity(stain[c1:c2, c3:c4].astype(np.float32), out_range=(0, 1))


def find_centers_and_crop(
    collection,
    isoform,
    cell_line,
    antibody,
    image_data,
    image_directory,
    debug=True,
    overwrite=False,
    save_root="cell_images",
    input_key="unmixed_lightfield",
    min_nuc_size = 100.0
):
    """
    This method finds the cell centers for all the cells in a larger brightfield image
    which pass the OTSU filter and minimum nuclear size. Creates cropped images around each cell
    and saves them as training data
    """
    def process_filename(filename):
        filepath = filename.split("images/")[-1]
        return os.path.join(image_directory, filepath)

    if input_key == "full_lightfield":
        outfolder = construct_singlecell_image_foldername(
            image_directory, isoform, folder="single_cells_crops"
        )
    else:
        outfolder = construct_singlecell_image_foldername(
            image_directory, isoform, folder="unmixed_single_cells_crops"
        )

    if not overwrite and len(image_data.get(save_root, [])) > 0:
        return
    if image_data is None or image_data.get(input_key) is None:
        print(
            f"skipping {outfolder} b/c scraped lightfield images are not set in mongo"
        )
        return

    if input_key == "full_lightfield" and (
        not (
            os.path.exists(
                image_data.get(input_key, {}).get("blue_red_green", "BAD_PATH")
            )
        )
        or not (os.path.exists(image_data.get(input_key, {}).get("yellow", "BAD_PATH")))
    ):
        print(f"skipping {outfolder} b/c full lightfield images are missing in system")
        return

    if input_key == "unmixed_lightfield" and (
        not (os.path.exists(image_data.get(input_key, {}).get("red", "BAD_PATH")))
        or not (os.path.exists(image_data.get(input_key, {}).get("blue", "BAD_PATH")))
        or not (os.path.exists(image_data.get(input_key, {}).get("green", "BAD_PATH")))
        or not (
            os.path.exists(
                image_data.get("full_lightfield", {}).get("yellow", "BAD_PATH")
            )
        )
    ):
        print(
            f"skipping {outfolder} b/c unmixed lightfield images are missing in system"
        )
        return

    print("working on ", outfolder)
    if not os.path.exists(outfolder):
        try:
            os.mkdir(outfolder)
        except:
            pass

    # Empty our previous images so we don't run into issues of adding the same element twice
    collection.update_one(
        filter={"_id": isoform["_id"]},
        update={
            "$pull": {
                f"cell_lines.$[elem1].antibodies.$[elem2].image_urls.$[elem3].{save_root}": {}
            }
        },
        array_filters=[
            {"elem1.name": cell_line["name"]},
            {"elem2.antibody_id": antibody["antibody_id"]},
            {"elem3.image_url": image_data["image_url"]},
        ],
    )

    # Get the image and resize
    def get_image_and_resize(filename, single_channel=False):
        image = Image.open(process_filename(filename))
        if single_channel:
            # In the case of a single channel image (e.g., Yellow Mitochondria staining)
            # We want to process the image as grayscale (reduce all color channels to one)
            image = np.array(ImageOps.grayscale(image)) / 255
        else:
            image = np.array(image)
        image_shape = image.shape[:2]
        image_shape = tuple(ti // 4 for ti in image_shape)
        return resize(image, image_shape), image_shape

    if input_key == "full_lightfield":
        # Split the rgb DAPI / Microtubule / Antibody image into channels
        three_channel_image, reference_shape = get_image_and_resize(
            image_data.get(input_key).get("blue_red_green")
        )
        microtubules_stain = three_channel_image[:, :, 0]
        antibody_stain = three_channel_image[:, :, 1]
        nuclei_stain = three_channel_image[:, :, 2]
    elif input_key == "unmixed_lightfield":
        # the stains are already separated for us in the unmixed case.
        microtubules_stain, reference_shape = get_image_and_resize(
            image_data.get(input_key).get("red"), single_channel=True
        )
        antibody_stain, reference_shape = get_image_and_resize(
            image_data.get(input_key).get("green"), single_channel=True
        )
        nuclei_stain, reference_shape = get_image_and_resize(
            image_data.get(input_key).get("blue"), single_channel=True
        )

    # Now handle the mitochondria only image
    mitochondria_stain, mito_image_shape = get_image_and_resize(
        image_data.get("full_lightfield").get("yellow"), single_channel=True
    )
    assert reference_shape == mito_image_shape

    # Segment the nuclear channel and get the nuclei
    val = threshold_otsu(nuclei_stain)
    smoothed_nuclei = gaussian(nuclei_stain, sigma=5.0)
    binary_nuclei = smoothed_nuclei > val
    binary_nuclei = remove_small_holes(binary_nuclei, area_threshold=300)
    labeled_nuclei = label(binary_nuclei)
    labeled_nuclei = clear_border(labeled_nuclei)
    labeled_nuclei = remove_small_objects(labeled_nuclei, min_size=min_nuc_size)

    # Iterate through each nuclei and get their centers (if the object is valid), and save to directory
    for i in range(1, np.max(labeled_nuclei)):
        current_nuc = labeled_nuclei == i
        if np.sum(current_nuc) > min_nuc_size:
            y, x = center_of_mass(current_nuc)
            x = int(x)
            y = int(y)

            # TODO: stop hard coding cropsize
            c1 = y - 128 // 2  # cropsize
            c2 = y + 128 // 2
            c3 = x - 128 // 2
            c4 = x + 128 // 2

            if c1 < 0 or c3 < 0 or c2 > reference_shape[0] or c4 > reference_shape[1]:
                pass
            else:
                nuclei_crop = process_patch(nuclei_stain, c1, c2, c3, c4)
                antibody_crop = process_patch(antibody_stain, c1, c2, c3, c4)
                mitochondria_crop = process_patch(mitochondria_stain, c1, c2, c3, c4)
                microtubule_crop = process_patch(microtubules_stain, c1, c2, c3, c4)

                outimagename = (
                    process_filename(image_data.get("full_lightfield").get("yellow"))
                    .rsplit("/", 1)[-1]
                    .rsplit("_", 3)[0]
                    + "_cell"
                    + str(i)
                )
                outimagepath = os.path.join(outfolder, outimagename)

                if (
                    os.path.exists(outimagepath + "_blue.tif") and overwrite
                ) or not os.path.exists(outimagepath + "_blue.tif"):
                    Image.fromarray(nuclei_crop).save(outimagepath + "_blue.tif")
                    Image.fromarray(antibody_crop).save(outimagepath + "_green.tif")
                    Image.fromarray(microtubule_crop).save(outimagepath + "_red.tif")
                    Image.fromarray(mitochondria_crop).save(
                        outimagepath + "_yellow.tif"
                    )

                if debug:
                    # For the sake of debugging also save as jpg
                    debug_outfolder = os.path.join(outfolder, "debug")
                    if not os.path.exists(debug_outfolder):
                        os.mkdir(debug_outfolder)
                    debug_outimagepath = os.path.join(debug_outfolder, outimagename)
                    if not os.path.exists(debug_outimagepath + "_blue.jpg"):
                        Image.fromarray(nuclei_crop * 255).convert("RGB").save(
                            debug_outimagepath + "_blue.jpg"
                        )
                        Image.fromarray(antibody_crop * 255).convert("RGB").save(
                            debug_outimagepath + "_green.jpg"
                        )
                        Image.fromarray(microtubule_crop * 255).convert("RGB").save(
                            debug_outimagepath + "_red.jpg"
                        )
                        Image.fromarray(mitochondria_crop * 255).convert("RGB").save(
                            debug_outimagepath + "_yellow.jpg"
                        )

                collection.update_one(
                    filter={"_id": isoform["_id"]},
                    update={
                        "$addToSet": {
                            f"cell_lines.$[elem1].antibodies.$[elem2].image_urls.$[elem3].{save_root}": {
                                "nuclei_channel": outimagepath + "_blue.tif",
                                "microtubule_channel": outimagepath + "_red.tif",
                                "antibody_channel": outimagepath + "_green.tif",
                                "mitochondria_channel": outimagepath + "_yellow.tif",
                            }
                        }
                    },
                    array_filters=[
                        {"elem1.name": cell_line["name"]},
                        {"elem2.antibody_id": antibody["antibody_id"]},
                        {"elem3.image_url": image_data["image_url"]},
                    ],
                )


def save_cropped_images(
    collection,
    splice_isoform_ids,
    image_directory,
    refresh_session,
    debug,
    num_isoform_images,
    overwrite=False,
    save_root="cell_images",
    input_key="unmixed_lightfield",
):
    if not os.path.exists(image_directory):
        os.mkdir(image_directory)

    splice_isoform_ids_subset = get_splice_variant_subset(
        splice_isoform_ids, num_isoform_images
    )

    cursor = collection.find(
        {"_id": {"$in": splice_isoform_ids_subset}}, no_cursor_timeout=True
    )
    for isoform in tqdm(
        cursor,
        total=len(splice_isoform_ids_subset),
    ):
        refresh_session()
        for cell_line in isoform.get("cell_lines", []):
            for antibody in cell_line.get("antibodies", []):
                for image_data in antibody.get("image_urls", []):
                    # It may become necessary to wrap this method call in a try / except block
                    # if you start seeing a lot of errors. Web scraping is an inherently fragile
                    # process so adding safety rails to fail gracefully can be helpful.
                    find_centers_and_crop(
                        collection=collection,
                        isoform=isoform,
                        cell_line=cell_line,
                        antibody=antibody,
                        image_data=image_data,
                        image_directory=image_directory,
                        debug=debug,
                        overwrite=overwrite,
                        save_root=save_root,
                        input_key=input_key,
                    )

    cursor.close()


def download_splice_isoforms(
    gene_collection,
    splice_isoform_collection,
    gene_ids,
    hpa_versions=["v22"],
):
    cursor = gene_collection.find({"_id": {"$in": gene_ids}}, no_cursor_timeout=True)
    for document in tqdm(
        cursor,
        total=gene_collection.count_documents({"_id": {"$in": gene_ids}}),
    ):
        if (
            splice_isoform_collection.count_documents(
                {"parent gene": document.get("Gene")}
            )
            == 0
        ):
            # If splice variants have not been scraped yet for a gene then scrape them
            output = []
            for version in hpa_versions:
                try:
                    output.append(antibody_scrape(document, version))
                except:
                    print("boo boo with version ", version)
                    pass
            try:
                output_df = pd.concat(output)
            except:
                continue

            def aggregation_func(group):
                return pd.Series(
                    {
                        "all_versions": tuple(group["version"]),
                        # Get the most recent splice isoform url
                        "splice_isoform_url": list(group["splice_isoform_url"])[-1],
                    }
                )

            splice_isoform_df = output_df.groupby(
                [
                    "splice_isoform_ensemble_id",
                    "splice_isoform_hpa_id",
                    "matching_antibodies",
                ],
                as_index=False,
            ).apply(aggregation_func)
            splice_isoform_df["_id"] = splice_isoform_df["splice_isoform_hpa_id"]
            splice_isoform_df["parent gene"] = document["Gene"]
            splice_isoform_df["parent gene name"] = document["Gene name"]
            try:
                splice_isoform_collection.insert_many(
                    splice_isoform_df.to_dict("records")
                )
            except:
                pass
    cursor.close()


def get_splice_variants_per_gene_set(
    gene_collection, gene_ids
):
    # Create an index on the "parent gene" field of the splice_isoform_collection
    # splice_isoform_collection.create_index("parent gene")

    # Use an aggregation pipeline to get all the splice variants for the genes in one query
    pipeline = [
        {"$match": {"_id": {"$in": gene_ids}}},
        {
            "$lookup": {
                "from": "splice_isoforms",
                "localField": "Gene",
                "foreignField": "parent gene",
                "as": "splice_variants",
            }
        },
        {"$unwind": "$splice_variants"},
        {"$group": {"_id": "$splice_variants._id"}},
    ]
    sv_ids = set([doc["_id"] for doc in gene_collection.aggregate(pipeline)])
    return list(sv_ids)


def get_data_by_genes(gene_ids, pool_size, hpa_versions):
    """
    This method:
     1. Downloads proteoforms (aka splice isoforms) for each gene in a gene_collection
     2. Downloads image urls for each antibody / cell_line available to the proteoform
     3. Scrapes localization labels specific to each antibody / cell line pair
    """
    # Important to define Mongo client separately for each thread!
    with MongoClient(
        host="localhost",
        port=27017,
        maxPoolSize=pool_size * 100,
        connectTimeoutMS=200000,
        serverSelectionTimeoutMS=200000,
    ) as client:
        splice_isoform_collection = client.hpa.splice_isoforms
        genes_collection = client.hpa.genes

        print("Downloading Splice Isoform IDs...")
        download_splice_isoforms(
            genes_collection, splice_isoform_collection, gene_ids, hpa_versions
        )

        print("Downloading Cell Lines / Antibodies / Image Urls...")
        download_cell_lines_antibodies_image_urls(
            genes_collection,
            splice_isoform_collection,
            gene_ids,
            hpa_versions,
            overwrite=True,
        )

        print("Scraping Subcellular Location Labels per Splice Variant/Cell Line")
        download_per_cell_line_subcell_labels(
            genes_collection, splice_isoform_collection, gene_ids, hpa_versions
        )


def get_data_by_splice_isoforms(
    sv_ids, pool_size, images_dir, debug, perform_unmixing=False, num_isoform_images=10000
):
    """
    This method:
     1. Scrapes proteoform sequences from Ensembl
     2. Calculate ESM-2 Embeddings for each proteoform
     3. Downloads lightfield images for each antibody / cell line combo available in HPA
     3b. Optionally performs spectral linear unmixing on the lightfield images
     4. Crops, preprocesses, and saves the individual cell images
    """
    # Important to define Mongo client separately for each thread!
    with MongoClient(
        host="localhost",
        port=27017,
        maxPoolSize=pool_size * 100,
        connectTimeoutMS=200000,
        serverSelectionTimeoutMS=200000,
    ) as client:
        with client.start_session(causal_consistency=True) as session:

            def refresh_session():
                client.admin.command(
                    "refreshSessions", [session.session_id], session=session
                )

            splice_isoform_collection = client.hpa.splice_isoforms

            print("Scraping Splice Isoform Sequences from Ensemble...")
            download_seqs_from_ensemble(splice_isoform_collection, sv_ids)

            print("Calculating Splice Isoform ESM2 Embeddings...")
            calculate_esm2_embeddings(splice_isoform_collection, sv_ids)

            print("Download Full Lightfield Images")
            download_images(
                splice_isoform_collection,
                sv_ids,
                images_dir,
                refresh_session,
                num_isoform_images=num_isoform_images,
                overwrite=False,
            )

            if perform_unmixing:
                print("Perform Linear Spectral Unmixing")
                unmix_spectra(
                    splice_isoform_collection,
                    sv_ids,
                    images_dir,
                    refresh_session,
                    num_isoform_images=num_isoform_images,
                    overwrite=False,
                )

            print("Crop & Save Single Cell Images")
            save_cropped_images(
                splice_isoform_collection,
                sv_ids,
                images_dir,
                refresh_session,
                debug=False, # This debug will make a JPG copy of each cell crop
                num_isoform_images=num_isoform_images,
                overwrite=False,
                save_root="cell_images" if not perform_unmixing else "unmixed_cell_images",
                input_key="full_lightfield" if not perform_unmixing else "unmixed_lightfield",
            )



def build_datasets(
    limit=None,
    # Subcell_loc_file should be downloaded from 
    # https://www.proteinatlas.org/download/subcellular_location.tsv.zip
    subcell_loc_file=get_data_path("subcellular_location.tsv"),
    images_dir=get_data_path("images"),
    pool_size=1,
    hpa_versions=[
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
    ],
    debug=True,
):
    print("You've specified this image directory! ", images_dir)
    global_client = MongoClient(maxPoolSize=500)
    global_genes_collection = global_client.hpa.genes

    print("Uploading genes into MongoDB database...")
    populate_genes(subcell_loc_file, global_genes_collection)
    gene_ids = global_genes_collection.find().distinct("_id")

    if not debug:
        random.seed(43)
        random.shuffle(gene_ids)

    if limit is None:
        limit = len(gene_ids)

    def chunks(l, total_size, num_threads, chunk_entirely=not (debug)):
        step_size = int(total_size / num_threads)
        for i in range(num_threads):
            start_idx = i * step_size
            if chunk_entirely and (i + 1) == num_threads:
                end_idx = len(l)
            else:
                end_idx = (i + 1) * step_size
            yield l[start_idx:end_idx]

    if debug:
        # If debugging then we want to only run one thread
        gene_id_chunks = list(chunks(gene_ids, limit, pool_size))
        get_data_by_genes(
            gene_id_chunks[0],
            pool_size=1,
            hpa_versions=hpa_versions,
        )
        global_splice_isoform_collection = global_client.hpa.splice_isoforms
        splice_isoform_ids = global_splice_isoform_collection.find().distinct("_id")

        get_data_by_splice_isoforms(
            sv_ids=splice_isoform_ids,
            pool_size=pool_size,
            images_dir=images_dir,
            debug=debug,
        )
    else:
        # If we are trying to go for efficiency then we can run multithreaded
        with multiprocessing.Pool(pool_size) as pool:
            pool.map(
                partial(
                    get_data_by_genes,
                    pool_size=pool_size,
                    hpa_versions=hpa_versions,
                ),
                list(chunks(gene_ids, limit, pool_size)),
            )
            pool.close()
            pool.join()
        global_splice_isoform_collection = global_client.hpa.splice_isoforms
        splice_isoform_ids = global_splice_isoform_collection.find().distinct("_id")

        if not debug:
            random.seed(43)
            random.shuffle(splice_isoform_ids)
        with multiprocessing.Pool(pool_size) as pool:
            pool.map(
                partial(
                    get_data_by_splice_isoforms,
                    pool_size=pool_size,
                    images_dir=images_dir,
                    debug=debug,
                ),
                list(chunks(splice_isoform_ids, len(splice_isoform_ids), pool_size)),
            )
            pool.close()
            pool.join()


if __name__ == "__main__":
    build_datasets(debug=True)