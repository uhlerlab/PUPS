import torch
import torch.nn as nn
from enum import Enum


class LayerStrategy(Enum):
    STATIC = 0
    # Halve the image dimensions
    POOL = 1
    # Double the image dimensions
    UPSAMPLE = 2


class SeparableConv2d(nn.Module):
    # Xception style Separable Conv from https://www.programmersought.com/article/1344745736/
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            padding=padding,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Inpainting_Model(nn.Module):
    def __init__(
        self,
        protein_encoding_dim,
        image_input_dim=128,
        image_input_channels=3,
        encoder_activation=nn.LeakyReLU,
        decoder_activation=nn.ReLU,
        conv_type=SeparableConv2d,
        residualize=True,
        residualization_method=torch.cat,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.residualize = residualize
        self.residualization_method = residualization_method
        self.instantiate_unet(
            input_dim=image_input_dim,
            input_channel_dim=image_input_channels,
            protein_encoding_dim=protein_encoding_dim,
            residualize=self.residualize,
            encoder_activation=encoder_activation,
            decoder_activation=decoder_activation,
        )

    def create_layer_list(
        self,
        input_dim,
        input_channel_dim,
        output_channel_dim,
        layer_strategy,
        activation,
        kernel_size=(3, 3),
    ):
        layer_list = [
            self.conv_type(
                in_channels=input_channel_dim,
                out_channels=output_channel_dim,
                kernel_size=kernel_size,
                padding=1,
            ),
            activation(),
            nn.BatchNorm2d(num_features=output_channel_dim),
        ]
        if layer_strategy == LayerStrategy.STATIC:
            output_dim = input_dim
        elif layer_strategy == LayerStrategy.POOL:
            layer_list += [nn.MaxPool2d(kernel_size=2, stride=2)]
            output_dim = input_dim / 2
        elif layer_strategy == LayerStrategy.UPSAMPLE:
            layer_list += [nn.Upsample(scale_factor=2)]
            output_dim = input_dim * 2

        return torch.nn.Sequential(*layer_list), output_dim

    def instantiate_unet(
        self,
        input_dim,
        input_channel_dim,
        protein_encoding_dim,
        residualize,
        encoder_activation,
        decoder_activation,
    ):
        """
        Pooling layers
        """
        # (N, 128, 128, 3) --> (N, 128, 128, 64)
        self.pool_layer1, image_dim = self.create_layer_list(
            input_dim=input_dim,
            input_channel_dim=input_channel_dim,
            output_channel_dim=64,
            layer_strategy=LayerStrategy.STATIC,
            activation=encoder_activation,
        )
        assert image_dim == 128
        # (N, 128, 128, 3) --> (N, 64, 64, 128)
        self.pool_layer2, image_dim = self.create_layer_list(
            input_dim=input_dim,
            input_channel_dim=64,
            output_channel_dim=128,
            layer_strategy=LayerStrategy.POOL,
            activation=encoder_activation,
        )
        assert image_dim == 64
        # (N, 64, 64, 128) --> (N, 32, 32, 256)
        self.pool_layer3, image_dim = self.create_layer_list(
            input_dim=image_dim,
            input_channel_dim=128,
            output_channel_dim=256,
            layer_strategy=LayerStrategy.POOL,
            activation=encoder_activation,
        )
        assert image_dim == 32
        # (N, 32, 32, 256) --> (N, 16, 16, 512)
        self.pool_layer4, image_dim = self.create_layer_list(
            input_dim=image_dim,
            input_channel_dim=256,
            output_channel_dim=512,
            layer_strategy=LayerStrategy.POOL,
            activation=encoder_activation,
        )
        assert image_dim == 16

        """
            Bottleneck layers
        """
        # (N, 16, 16, 512) --> (N, 16, 16, 512)
        self.bottleneck_layer1, image_dim = self.create_layer_list(
            input_dim=image_dim,
            input_channel_dim=512,
            output_channel_dim=512,
            layer_strategy=LayerStrategy.STATIC,
            activation=encoder_activation,
        )
        assert image_dim == 16
        # (N, 16, 16, 512 + encoding_dim) --> (N, 16, 16, 512 + encoding_dim / 2)
        self.bottleneck_layer2, image_dim = self.create_layer_list(
            input_dim=image_dim,
            input_channel_dim=512 + protein_encoding_dim,
            output_channel_dim=512 + int(protein_encoding_dim / 2),
            layer_strategy=LayerStrategy.STATIC,
            activation=encoder_activation,
        )
        assert image_dim == 16
        # (N, 16, 16, 512 + encoding_dim / 2) --> (N, 16, 16, 512)
        self.bottleneck_layer3, image_dim = self.create_layer_list(
            input_dim=image_dim,
            input_channel_dim=512 + int(protein_encoding_dim / 2),
            output_channel_dim=512,
            layer_strategy=LayerStrategy.STATIC,
            activation=encoder_activation,
        )
        assert image_dim == 16

        """
            Upsampling layers
        """
        # (N, 16, 16, 512) --> (N, 32, 32, 256)
        self.upsample_layer1, image_dim = self.create_layer_list(
            input_dim=image_dim,
            input_channel_dim=512 if not residualize else 512 * 2,
            output_channel_dim=256,
            layer_strategy=LayerStrategy.UPSAMPLE,
            activation=decoder_activation,
        )
        assert image_dim == 32
        # (N, 32, 32, 256) --> (N, 64, 64, 128)
        self.upsample_layer2, image_dim = self.create_layer_list(
            input_dim=image_dim,
            input_channel_dim=256 if not residualize else 256 * 2,
            output_channel_dim=128,
            layer_strategy=LayerStrategy.UPSAMPLE,
            activation=decoder_activation,
        )
        assert image_dim == 64
        # (N, 64, 64, 128) --> (N, 128, 128, 64)
        self.upsample_layer3, image_dim = self.create_layer_list(
            input_dim=image_dim,
            input_channel_dim=128 if not residualize else 128 * 2,
            output_channel_dim=64,
            layer_strategy=LayerStrategy.UPSAMPLE,
            activation=decoder_activation,
        )
        assert image_dim == 128
        # (N, 128, 128, 64) --> (N, 128, 128, 1)
        self.upsample_layer4, image_dim = self.create_layer_list(
            input_dim=image_dim,
            input_channel_dim=64 if not residualize else 64 * 2,
            output_channel_dim=1,
            layer_strategy=LayerStrategy.STATIC,
            activation=decoder_activation,
        )
        assert image_dim == 128

    def forward(self, cell_in, protein_in):
        """
        image_input_channels
        Args:
            cell_in: [batch_size, image_input_dim, image_input_dim, image_input_channels] tensor of landmark stain 
            protein_in: [batch_size, protein_encoding_dim] tensor of protein latent representation 
        Returns:
            regression: [batch_size, image_input_dim, image_input_dim, 1] tensor of predicted target protein stain
        """
        out1 = self.pool_layer1(cell_in)
        out2 = self.pool_layer2(out1)
        out3 = self.pool_layer3(out2)
        out4 = self.pool_layer4(out3)

        out5 = self.bottleneck_layer1(out4)
        protein_in_repeated = (
            protein_in.unsqueeze(-1).unsqueeze(-1).repeat((1, 1, 16, 16))
        )
        stacked = self.residualization_method((out5, protein_in_repeated), 1)
        out6 = self.bottleneck_layer2(stacked)
        out7 = self.bottleneck_layer3(out6)

        if self.residualize:
            out7 = self.residualization_method((out7, out4), 1)
        out8 = self.upsample_layer1(out7)
        if self.residualize:
            out8 = self.residualization_method((out8, out3), 1)
        out9 = self.upsample_layer2(out8)
        if self.residualize:
            out9 = self.residualization_method((out9, out2), 1)
        out10 = self.upsample_layer3(out9)
        if self.residualize:
            out10 = self.residualization_method((out10, out1), 1)
        out11 = self.upsample_layer4(out10)
        return out11
