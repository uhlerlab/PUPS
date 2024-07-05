
import torch
import torch.nn as nn

class SeparableConv1d(nn.Module):
    # Xception style Separable Conv adapted from https://www.programmersought.com/article/1344745736/
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False, padding=1):
        super(SeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            stride=stride,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            padding=padding,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class LightAttentionNN(nn.Module):
    # Adapted from Hannes et al. https://academic.oup.com/bioinformaticsadvances/article/1/1/vbab035/6432029
    # But with Xception style convolution, ESM2 representation, and upated parameterization.

    def __init__(
        self,
        embeddings_dim=1280,
        dropout=0.10,
        kernel_size=15,
        conv_dropout: float = 0.10,
        num_conv_layers=1,
        final_mlp_dim=300,
        conv_type=SeparableConv1d,
    ):
        assert (
            kernel_size % 2 == 1
        ), "Kernel size has to be odd for padding to work out..."
        super(LightAttentionNN, self).__init__()
        self.feature_convolution = self.create_layer_list(
            embeddings_dim=embeddings_dim,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            conv_type=conv_type,
        )
        self.attention_convolution = self.create_layer_list(
            embeddings_dim=embeddings_dim,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            conv_type=conv_type,
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)
        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, final_mlp_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(final_mlp_dim),
        )

    def create_layer_list(
        self,
        embeddings_dim,
        num_conv_layers,
        kernel_size,
        stride=1,
        activation=nn.ReLU,
        conv_type=nn.Conv1d,
    ):
        layer_list = []
        for i in range(num_conv_layers):
            layer_list.append(
                conv_type(
                    embeddings_dim,
                    embeddings_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layer_list.append(activation())
            layer_list.append(torch.nn.BatchNorm1d(embeddings_dim))
        return torch.nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor, x_lens, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 1, sequence_length, embeddings_dim] embedding tensor that should be classified
            x_lens: [batch_size, sequence_length] peptide lengths of x
        Returns:
            classification: [batch_size, output_dim] tensor with logits
        """
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = torch.transpose(x, 1, 2)
        o = self.feature_convolution(x)
        o = self.dropout(o)
        attention = self.attention_convolution(x)

        sequence_lengths = x.shape[2]
        mask = torch.arange(sequence_lengths)[None, :].type_as(x_lens) < x_lens[:, None]
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)
        o1 = torch.sum(o * self.softmax(attention), dim=-1)
        o2, _ = torch.max(o, dim=-1)
        o = torch.cat([o1, o2], dim=-1)
        return self.linear(o)
