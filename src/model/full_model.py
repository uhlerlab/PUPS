import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))
print(str(pathlib.Path(__file__).parent.parent.absolute().parent))

# NOTE: Need to import tensorflow before pytorch lightning else protocol buffer runtime library miscongruencies
import tensorflow as tf  
import torch
from torch import optim
import numpy as np
import lightning.pytorch as pl
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.dataset.dataset import CLASSES
from src.model.nn_light_attention import LightAttentionNN
from src.model.nn_multilabel_mlp import SimpleMLPNN
from src.model.nn_unet import Inpainting_Model

class SubCellProtModel(pl.LightningModule):
    def __init__(
        self,
        intermediate_layer_size=300,
        multilabel_weight=1,
        embeddings_dim=1280
    ):
        """
            intermediate_layer_size: dimension of protein representation passed from Light Attention to U-Net & mulitilabel_mlp
            multilabel_weight: multiplier for the mulitlabel loss relative to the unet reconstruction loss
            embeddings_dim: dimension of the input (ESM2) representations
        """
        super().__init__()
        torch.cuda.empty_cache()

        self.intermediate_layer_size = intermediate_layer_size
        self.multilabel_weight = multilabel_weight

        self.light_attention_trunk = LightAttentionNN(
            embeddings_dim=embeddings_dim,
            final_mlp_dim=self.intermediate_layer_size,
        )
        self.multilabel_mlp = SimpleMLPNN(
            input_dim=self.intermediate_layer_size, output_dim=len(CLASSES[0])
        )
        self.cell_inpainting_unet = Inpainting_Model(
            protein_encoding_dim=self.intermediate_layer_size
        )

        self.multilabel_loss_func = BCEWithLogitsLoss()
        self.inpainting_loss_func = torch.nn.MSELoss()

    def call_model(self, X_esm2_encoding, X_protein_len, X_landmark_stains):
        X_esm2_encoding = X_esm2_encoding.to(self.device)
        X_protein_len = X_protein_len.to(self.device)
        X_landmark_stains = X_landmark_stains.to(self.device)
        protein_latent_rep = self.light_attention_trunk(
            X_esm2_encoding,
            X_protein_len,
        )

        y_pred_antibody_stain = self.cell_inpainting_unet(
            X_landmark_stains,
            protein_latent_rep,
        )
        y_pred_multilabel = self.multilabel_mlp(protein_latent_rep)
        return y_pred_antibody_stain, y_pred_multilabel
    

    def loop_step(self, batch, stage):
        """
        Calculate the combined loss and individual losses
        """
        (
            X_esm2_encoding,
            X_protein_len,
            X_landmark_stains,
            y_multilabel,
            y_antibody_stain,
        ) = batch
        y_pred_antibody_stain, y_pred_multilabel = self.call_model(
            X_esm2_encoding, X_protein_len, X_landmark_stains
        )
        multilabel_loss = self.multilabel_loss_func(
            y_pred_multilabel,
            y_multilabel.type_as(y_pred_multilabel),
        )
        inpainting_loss = self.inpainting_loss_func(
            y_pred_antibody_stain.reshape(
                y_pred_antibody_stain.shape[0],
                y_pred_antibody_stain.shape[2],
                y_pred_antibody_stain.shape[3],
            ),
            y_antibody_stain.type_as(y_pred_antibody_stain),
        )
        multilabel_loss = self.multilabel_weight * multilabel_loss
        combined_loss = multilabel_loss + inpainting_loss

        """
            Log
        """
        self.log(
            f"{stage}_multilabel_loss",
            multilabel_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_inpainting_loss",
            inpainting_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_combined_loss",
            combined_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return combined_loss

    def training_step(self, batch, batch_idx):
        self.light_attention_trunk.train()
        self.multilabel_mlp.train()
        self.cell_inpainting_unet.train()
        return self.loop_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.light_attention_trunk.eval()
        self.multilabel_mlp.eval()
        self.cell_inpainting_unet.eval()
        return self.loop_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.light_attention_trunk.eval()
        self.multilabel_mlp.eval()
        self.cell_inpainting_unet.eval()
        return self.loop_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        self.light_attention_trunk.eval()
        self.multilabel_mlp.eval()
        self.cell_inpainting_unet.eval()
        (
            X_esm2_encoding,
            X_protein_len,
            X_landmark_stains,
            _y_multilabel,
            _y_antibody_stain,
        ) = batch
        y_pred_antibody_stain, y_pred_multilabel = self.call_model(
            X_esm2_encoding, X_protein_len, X_landmark_stains
        )
        y_pred_ranked = torch.sigmoid(y_pred_multilabel).cpu().detach().numpy()
        y_pred = np.round(y_pred_ranked)
        return y_pred_antibody_stain, y_pred, y_pred_ranked

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": opt,
            "scheduler": ReduceLROnPlateau(opt, patience=2),
            "monitor": "val_combined_loss",
        }

    def backward(self, loss):
        loss.backward()


