import seaborn as sns
import matplotlib as plt

import tensorflow as tf
import torch
from src.dataset.dataset import DatasetType
from src.utils.data_handling_utils import Retrieval_Data
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def get_rby_images(
    cell_idx, train_dataset, test_dataset, val_dataset, filter_low_values=None
):
    """
    Fetches the microtubule (red), nucleui (blue), and endoplasmic reticulum (yellow) stains
    """
    dataset_name, idx = cell_idx.split("_")[0], int(cell_idx.split("_")[1])
    if dataset_name.lower() == DatasetType.TRAIN.name.lower():
        dataset = train_dataset
    elif dataset_name.lower() == DatasetType.TEST.name.lower():
        dataset = test_dataset
    elif dataset_name.lower() == DatasetType.EVAL.name.lower():
        dataset = val_dataset

    (
        _cell_X_esm2_encoding_orig,
        _cell_X_protein_len_orig,
        X_landmark_stains,
        _cell_y_multilabel_orig,
        _cell_y_antibody_stain_orig,
    ), _cell_line_metadata = dataset.get_item_verbose(
        idx, filter_low_values=filter_low_values
    )

    nuclei_img = np.zeros(X_landmark_stains.T.shape)
    nuclei_img[:, :, 2] = X_landmark_stains[0, :, :]
    nuclei_img = Image.fromarray(np.uint8(np.abs(nuclei_img) * 255), mode="RGB")

    microtubule_img = np.zeros(X_landmark_stains.T.shape)
    microtubule_img[:, :, 0] = X_landmark_stains[1, :, :]
    microtubule_img = Image.fromarray(
        np.uint8(np.abs(microtubule_img) * 255), mode="RGB"
    )

    mitochondria_img = np.zeros(X_landmark_stains.T.shape)
    mitochondria_img[:, :, 0] = X_landmark_stains[2, :, :]
    mitochondria_img[:, :, 1] = X_landmark_stains[2, :, :]
    mitochondria_img = Image.fromarray(
        np.uint8(np.abs(mitochondria_img) * 255), mode="RGB"
    )
    return microtubule_img, nuclei_img, mitochondria_img


def viz_matrix_on_fly(
    loaded_model,
    cell_idxs,
    isoform_idxs,
    get_data,
    train_dataset,
    test_dataset,
    val_dataset,
    show_landmark_stains=True,
    show_y_true_on_diagonal=False,
    figsize=(20, 20),
):
    if show_y_true_on_diagonal:
        assert (
            cell_idxs == isoform_idxs
        ), "Only can show y_true on the diagonal if rows & columns are the same"
    fig, axes = plt.subplots(
        nrows=len(cell_idxs),
        ncols=len(isoform_idxs) + (3 if show_landmark_stains else 0),
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    if show_landmark_stains:
        for row_idx, cell_idx in enumerate(cell_idxs):
            microtubule_img, nuclei_img, mitochondria_img = get_rby_images(
                cell_idx, train_dataset, test_dataset, val_dataset
            )
            axes[row_idx, 0].imshow(nuclei_img)
            axes[row_idx, 1].imshow(microtubule_img)
            axes[row_idx, 2].imshow(mitochondria_img)
            if row_idx == 0:
                axes[row_idx, 0].set_title("Nucleus", fontsize=12)
                axes[row_idx, 1].set_title("Microtubule", fontsize=12)
                axes[row_idx, 2].set_title("Endoplasmic Ret.", fontsize=12)

    for row_idx, cell_idx in enumerate(cell_idxs):
        for col_idx, isoform_idx in enumerate(isoform_idxs):
            ax = axes[row_idx, col_idx + (3 if show_landmark_stains else 0)]

            if show_y_true_on_diagonal and row_idx == col_idx:
                _y_multilabel, y_antibody_stain = get_data(
                    isoform_idx, Retrieval_Data.TRUE_LABELS
                )
                image_data = y_antibody_stain.squeeze()
                y_true_img = np.zeros((image_data.shape[0], image_data.shape[1], 3))
                y_true_img[:, :, 1] = image_data
                y_true_img = Image.fromarray(
                    np.uint8(np.abs(y_true_img) * 255), mode="RGB"
                )
                ax.imshow(y_true_img)
            else:
                X_landmark_stains = get_data(cell_idx, Retrieval_Data.CELL_IMAGE)
                X_esm2_encoding, X_protein_len = get_data(
                    isoform_idx, Retrieval_Data.PROTEIN_SEQ
                )
                (
                    y_pred_antibody_stain,
                    _y_pred_multilabel,
                    _y_pred_ranked,
                ) = loaded_model.predict_step(
                    (
                        X_esm2_encoding.unsqueeze(0),
                        torch.Tensor([X_protein_len]),
                        torch.from_numpy(X_landmark_stains).unsqueeze(0),
                        None,
                        None,
                    ),
                    batch_idx=0,
                )
                image_data = y_pred_antibody_stain.detach().numpy().squeeze()
                img = np.uint8(np.abs(image_data) * 255)
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    # Add cell line names as y-labels
    for row_idx, cell_idx in enumerate(cell_idxs):
        ax = axes[row_idx, 0]
        metadata = get_data(cell_idx, Retrieval_Data.METADATA)
        ax.set_ylabel(metadata["cell_line"], fontsize=12, rotation=0, labelpad=20)

    # Add splice isoform names as x-labels
    for col_idx, isoform_idx in enumerate(isoform_idxs):
        ax = axes[0, col_idx + (3 if show_landmark_stains else 0)]
        metadata = get_data(isoform_idx, Retrieval_Data.METADATA)
        ax.set_title(metadata["splice_isoform_id"], fontsize=12)

    # Hide all axis labels & tick marks
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.tick_params(axis="both", length=0)

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show()


def plot_projection_with_cuts(
    plotting_df,
    x_lab,
    y_lab,
    hue_lab="cell_line",
    x_cut=(1500, 33000),
    y_cut=(200, 3500),
    x_margin=50,
    y_margin=10,
):
    hue_order = sorted(set(plotting_df[hue_lab]))
    f, axes = plt.subplots(ncols=2, nrows=2)
    f.set_figheight(6)
    f.set_figwidth(8)
    ax3, ax4, ax1, ax2 = axes.flat

    def create_scatterplot(axis):
        return sns.scatterplot(
            data=plotting_df,
            x=x_lab,
            y=y_lab,
            style=hue_lab,
            hue=hue_lab,
            hue_order=hue_order,
            ax=axis,
        )

    ax = create_scatterplot(ax1)
    ax = create_scatterplot(ax2)
    ax = create_scatterplot(ax3)
    ax = create_scatterplot(ax4)

    x_min, x_max, y_min, y_max = (
        plotting_df[x_lab].min() - x_margin,
        plotting_df[x_lab].max() + x_margin,
        plotting_df[y_lab].min() - y_margin,
        plotting_df[y_lab].max() + y_margin,
    )

    # Set y/x axes cuts
    ax1.set_xlim(x_min, x_cut[0])
    ax1.set_ylim(y_min, y_cut[0])
    ax2.set_xlim(x_cut[1], x_max)
    ax2.set_ylim(y_min, y_cut[0])
    ax3.set_xlim(x_min, x_cut[0])
    ax3.set_ylim(y_cut[1], y_max)
    ax4.set_xlim(x_cut[1], x_max)
    ax4.set_ylim(y_cut[1], y_max)

    # Hide splines
    ax3.spines["bottom"].set_visible(False)
    ax4.spines["bottom"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Hide X and Y axes label marks
    ax3.xaxis.set_tick_params(labelbottom=False)
    ax3.set_xticks([])
    ax3.set_xlabel("")
    ax4.xaxis.set_tick_params(labelbottom=False)
    ax4.set_xticks([])
    ax4.set_xlabel("")
    ax2.yaxis.set_tick_params(labelleft=False)
    ax2.set_yticks([])
    ax2.set_ylabel("")
    ax4.yaxis.set_tick_params(labelleft=False)
    ax4.set_yticks([])
    ax4.set_ylabel("")

    [[c.get_legend().remove() for c in r] for r in axes]
    handles, labels = ax4.get_legend_handles_labels()
    f.legend(handles, labels, loc="upper right")
    sns.move_legend(f, "upper left", bbox_to_anchor=(1, 1))
    plt.show()

