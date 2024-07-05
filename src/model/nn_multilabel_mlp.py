import torch
import torch.nn as nn
import numpy as np

class SimpleMLPNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super().__init__()
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, protein_in: torch.Tensor) -> torch.Tensor:
        return self.output(protein_in)

    def predict(self, protein_in: torch.Tensor):
        self.eval()
        x = self.forward(protein_in)
        y_pred_ranked = torch.sigmoid(x).cpu().detach().numpy()
        y_pred = np.round(y_pred_ranked)
        return y_pred, y_pred_ranked
