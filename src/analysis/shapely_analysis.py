from src.model.full_model import SubCellProtModel
from src.utils.data_handling_utils import initialize_datasets, Retrieval_Data
from src.utils.batch_run_utils import get_isoforms_of_interest
import numpy as np
from src.dataset.dataset import CLASSES
from tqdm import tqdm
from numpy import savetxt
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import shap
import torch
from lightning.pytorch import seed_everything
from tqdm.notebook import tqdm_notebook

tqdm_notebook()

MAX_PEPTIDE_LEN = 2000
AA_FEATURE_DIM = 1280


def shapley_analysis_sliding_kernel_explainer(
    model,
    baseline_Xs,
    target_X,
    target_x_lens,
    target_X_landmark_stains,
    kernel_size=5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    nsamples=200,
    batch_sizes=100,  # adjust this parameter lower if you hit the memory limit
):
    """
    This expects batch composed of:
    * baseline_Xs in shape (N, 2000, 1280) where N is the number of background samples
    * target_X in shape (1, 2000, 1280)
    * target_x_lens in shape (1)
    * target_X_landmark_stains in shape (3, 128, 128)
    """

    np.random.seed(0)
    seed_everything(0)
    baseline_Xs = torch.Tensor(baseline_Xs).to(device)
    target_X = torch.Tensor(target_X).to(device)
    target_x_lens = torch.Tensor(target_x_lens).to(device)
    target_X_landmark_stains = torch.Tensor(target_X_landmark_stains).to(device)

    def f(bitmaps):
        def chunk_list(lst, chunk_size=batch_sizes):
            return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

        y_pred_ranked_list = []
        # We need to go through this as chunks since our representations are too large
        for bitmap in tqdm(chunk_list(bitmaps), desc="Processing bitmaps"):
            X = decode_bitmap(bitmap)
            _y_pred_antibody_stain, _y_pred, y_pred_ranked = model.predict_step(
                (
                    X.unsqueeze(1),
                    target_x_lens.repeat(X.shape[0]),
                    target_X_landmark_stains.unsqueeze(0).repeat((X.shape[0], 1, 1, 1)),
                    None,
                    None,
                ),
                batch_idx=0,
            )
            y_pred_ranked_list.append(y_pred_ranked)
        return np.vstack(y_pred_ranked_list)

    def decode_bitmap(encoded_bitmaps):
        def helper(bitmap):
            """
            Now go through the encoded bitmap and reconstruct X from the bitmap
            Again, -1 means take from the target representation,
            0 means from the first baseline representation,
            1 means from the second baseline representation,
            ...
            N means from the Nth baseline representation.
            """
            residue_chunks = []
            for idx, residue_chunk_bit in enumerate(bitmap):
                if residue_chunk_bit == -1:
                    residue_chunks.append(
                        np.array(
                            target_X[0][
                                idx * kernel_size : (idx + 1) * kernel_size
                            ].cpu()
                            if isinstance(target_X, torch.Tensor)
                            else target_X[0][
                                idx * kernel_size : (idx + 1) * kernel_size
                            ]
                        )
                    )
                else:
                    baseline_X = baseline_Xs[residue_chunk_bit]
                    residue_chunks.append(
                        np.array(
                            baseline_X[0][
                                idx * kernel_size : (idx + 1) * kernel_size
                            ].cpu()
                            if isinstance(baseline_X, torch.Tensor)
                            else baseline_X[0][
                                idx * kernel_size : (idx + 1) * kernel_size
                            ]
                        )
                    )
            return np.vstack(residue_chunks)

        decoded = np.array([helper(bitmap) for bitmap in encoded_bitmaps])
        return torch.Tensor(decoded).to(device)

    def encode_bitmap(identifier=-1):
        """
        We are simplifying the encoding of each protein since otherwise the feature space is too large
        Every protein will be represented by a bitmap of length protein_len / kernel_size (so we're grouping
        multiple residues into each feature. Each position of the bitmap uses a identifier

        -1 means we take the TARGET embedding,
        0 means we take the FIRST baseline embedding,
        1 means we take the SECOND baseline embedding,
        ...
        N means we take the Nth baseline residue embedding.
        """

        bitmap_shape = np.ceil(MAX_PEPTIDE_LEN / kernel_size)
        target_protein_dim = int(
            np.squeeze(np.ceil(target_x_lens.cpu() / kernel_size)).item()
        )
        return np.concatenate(
            (
                np.repeat(identifier, target_protein_dim),
                # Cut all baseline protein embeddings down to target protein length
                np.repeat(-1, int(bitmap_shape - target_protein_dim)),
            )
        )

    target_bitmap = np.array([encode_bitmap(identifier=-1)])
    reconstructed_X = decode_bitmap(target_bitmap)
    assert (
        target_X == reconstructed_X
    ).all(), "The bit map de/construction does not work"
    summary = np.stack(
        [
            encode_bitmap(identifier=baseline_X_identifier)
            for baseline_X_identifier in range(len(baseline_Xs))
        ]
    )
    explainer = shap.KernelExplainer(f, summary)

    # Then use "num_perturbation_samples" perterbation samples to estimate the SHAP values for a given prediction
    # Note that this requires (num_background_samples * num_perturbation_samples) evaluations of the model.
    encoded_shap_values = explainer.shap_values(target_bitmap, nsamples=nsamples)

    return encoded_shap_values, explainer.expected_value


def perform_shapely(
    isoforms_for_analysis,
    collection_name,
    model_checkpoint,
    num_baseline_isoforms=300,  # Number of baseline proteins to compare against
    nsamples=1000,  # Number of samples to take per baseline protein
    sliding_kernel_size=10,  # Since our feature space is so large, we want to group multiple residues together as one feature. The sliding kernel groups X number of residues together.
):
    train_dataset, val_dataset, test_dataset, get_data = initialize_datasets(
        collection_name
    )

    loaded_model = SubCellProtModel().load_from_checkpoint(
        model_checkpoint,
        collection_name=collection_name,
        batch_size=32,
    )

    # Construct baseline X averaged
    print("gather baseline isoforms")
    baseline_isoforms = get_isoforms_of_interest(
        collection_name=collection_name,
        total_investigated_isoforms=num_baseline_isoforms,
        get_data=get_data,
        seed=0,
    )

    def perform_shap_analysis(
        target_isoform_id,
        model,
        baseline_Xs,
        target_X,
        target_x_lens,
        target_X_landmark_stains,
        nsamples=200,
        kernel_size=5,
    ):
        assert len(target_X) == 1, (
            "For now this method can only handle single protein targets"
            / " since we have to are getting creative with encoding/decoding bit maps"
        )
        model.eval()

        res = shapley_analysis_sliding_kernel_explainer(
            model=model,
            baseline_Xs=baseline_Xs,
            target_X=target_X,
            target_x_lens=target_x_lens,
            target_X_landmark_stains=target_X_landmark_stains,
            kernel_size=kernel_size,
            nsamples=nsamples,
        )

        expected_vals = res[1]
        print("expected vals for the different compartments: ", expected_vals)
        compartment_shap_vals = [shap_vals for shap_vals in res[0]]
        [
            savetxt(
                f"results/shapely_results/{target_isoform_id}_{CLASSES[0][compartment_idx]}_({compartment_idx}).csv",
                comp_shap,
                delimiter=",",
            )
            for compartment_idx, comp_shap in enumerate(compartment_shap_vals)
        ]
        return expected_vals

    expected_vals = []
    for isoform_id in isoforms_for_analysis:
        X_investigation, x_len_investigation = get_data(
            isoform_id, retrieval_data=Retrieval_Data.PROTEIN_SEQ
        )
        X_landmark_stains = get_data(
            isoform_id, retrieval_data=Retrieval_Data.CELL_IMAGE
        )

        filtered_baseline_isoforms = []
        full_baseline_isoforms = []
        averaged_baseline_X = torch.zeros(X_investigation.shape)

        print("target isoform len is ", x_len_investigation)
        for isoform in tqdm(baseline_isoforms):
            X, x_len = get_data(
                isoform.split(" ")[1], retrieval_data=Retrieval_Data.PROTEIN_SEQ
            )
            if x_len < x_len_investigation:
                # Skipping all isoforms which are smaller than our target isoform in our baseline
                # Because otherwise we can't "replace" the amino acid at end of our target sequence
                # with a residue from the baseline isoform.
                continue
            averaged_baseline_X += X
            filtered_baseline_isoforms.append(isoform.split(" ")[1])
            full_baseline_isoforms.append(X)

        full_baseline_isoforms = np.stack(full_baseline_isoforms)
        averaged_baseline_X = averaged_baseline_X / len(filtered_baseline_isoforms)
        print(
            f"total samples in our baseline (that pass the length check): {len(filtered_baseline_isoforms)}"
        )

        expected_vals.append(
            perform_shap_analysis(
                target_isoform_id=isoform_id,
                model=loaded_model,
                baseline_Xs=np.stack([averaged_baseline_X]),
                target_X=X_investigation,
                target_x_lens=[x_len_investigation],
                target_X_landmark_stains=X_landmark_stains,
                nsamples=nsamples,
                kernel_size=sliding_kernel_size,
            )
        )

    return expected_vals


def viz_shapely(
    target_isoform_id,
    collection_name,
    model_checkpoint,
    sliding_kernel_size=10,
    compartments=CLASSES[0],
):
    train_dataset, val_dataset, test_dataset, get_data = initialize_datasets(
        collection_name
    )
    loaded_model = SubCellProtModel().load_from_checkpoint(
        model_checkpoint,
        collection_name=collection_name,
        batch_size=32,
    )
    X_investigation, x_len_investigation = get_data(
        target_isoform_id, retrieval_data=Retrieval_Data.PROTEIN_SEQ
    )
    X_landmark_stains = get_data(
        target_isoform_id, retrieval_data=Retrieval_Data.CELL_IMAGE
    )

    def viz_shapely_helper(
        compartment_name,
        x_len_investigation,
        ax,
        sigma=0.5,
        x_axis_title="Residue Index",
    ):
        compartment_idx = CLASSES[0].index(compartment_name)
        filename = f"results/shapely_results/{target_isoform_id}_{compartment_name}_({compartment_idx}).csv"
        print(filename)
        shap_values = pd.read_csv(filename, header=None).to_numpy()[0]

        def gaussian_kernel_1d(sigma):
            kernel_radius = np.ceil(sigma) * 3
            kernel_size = kernel_radius * 2 + 1
            ax = np.arange(-kernel_radius, kernel_radius + 1.0, dtype=np.float32)
            kernel = np.exp(-(ax**2) / (2.0 * sigma**2))
            return (kernel / np.sum(kernel)).reshape(1, kernel.shape[0])

        gaussian = np.ones(10) / 10  # gaussian_kernel_1d(sigma)[0]
        gaussian_sliding_window = np.convolve(shap_values, gaussian, mode="full")

        target_x_len = int(x_len_investigation / sliding_kernel_size)
        x_vals = [sliding_kernel_size * x for x in list(range(target_x_len))]
        ax.bar(x_vals, shap_values[:target_x_len], width=sliding_kernel_size)
        ax.plot(
            x_vals,
            gaussian_sliding_window[:target_x_len],
            "red",
        )
        ax.set_title(str(compartment_name))
        ax.set_xlabel(x_axis_title)

    fig, axs = plt.subplots(int(np.ceil(len(compartments) / 2)), 2, figsize=(12, 16))
    fig.tight_layout(pad=3.0)  # Adjust spacing between subplots

    # Iterate over compartments and corresponding subplot
    for compartment, ax in zip(compartments, axs.flatten()):
        viz_shapely_helper(
            compartment_name=compartment,
            x_len_investigation=x_len_investigation,  # Adjust as needed
            ax=ax,
            x_axis_title="Residue Index",
        )

    (
        _y_pred_antibody_stain,
        y_pred_multilabel,
        _y_pred_multilabel_raw,
    ) = loaded_model.predict_step(
        (
            X_investigation.unsqueeze(0),
            torch.Tensor([x_len_investigation]),
            torch.Tensor(X_landmark_stains).unsqueeze(0),
            None,
            None,
        ),
        batch_idx=0,
    )
    assert len(CLASSES[0]) == len(y_pred_multilabel[0])
    pred_location_labels = [
        compartment
        for compartment, y_pred in zip(CLASSES[0], y_pred_multilabel[0])
        if y_pred
    ]
    metadata = get_data(target_isoform_id, retrieval_data=Retrieval_Data.METADATA)
    fig.text(
        0.02,
        0.5,
        "Positional Shapely Value",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    plt.suptitle(
        f"Shapely Analysis for {metadata['splice_isoform_id']} \n(True: {metadata['location_labels']}, Pred: {pred_location_labels})"
    )
    plt.subplots_adjust(top=0.92, left=0.10)

    plt.show()
