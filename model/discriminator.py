from typing import Tuple

import torch
from torch.nn import Module, Linear, LeakyReLU, Conv3d, BatchNorm3d, Sequential, Sigmoid, AdaptiveAvgPool3d


class Discriminator(Module):
    def __init__(self,
                 label_dim: int = 1) -> None:
        """
        The discriminator Module hardcoded to take a one channel 20 dimensional voxel grid.

        :param label_dim: The dimension of the property. Should mostly be 1
        """
        super(Discriminator, self).__init__()

        self.label_condition_disc = Linear(label_dim, 20 * 20 * 20)
        self.fc = Linear(512, 1)
        self.sig = Sigmoid()
        self.model = Sequential(Conv3d(in_channels=1, out_channels=4, kernel_size=(2, 2, 2), stride=1, bias=False),
                                LeakyReLU(0.2, inplace=True),
                                Conv3d(in_channels=4, out_channels=8, kernel_size=(2, 2, 2), stride=1, bias=False),
                                BatchNorm3d(8, momentum=0.1, eps=0.8),
                                LeakyReLU(0.2, inplace=True),
                                Conv3d(in_channels=8, out_channels=4, kernel_size=(2, 2, 2), stride=1, bias=False),
                                BatchNorm3d(4, momentum=0.1, eps=0.8),
                                LeakyReLU(0.2, inplace=True),
                                Conv3d(in_channels=4, out_channels=1, kernel_size=(2, 2, 2), stride=1, bias=False),
                                LeakyReLU(0.2, inplace=True),
                                AdaptiveAvgPool3d(output_size=(512, 1, 1))
                                )  # average pool makes reshaping easy, but not ideal architectural choice

    def forward(self,
                inputs: Tuple[torch.tensor, torch.tensor] = None) -> torch.tensor:
        """
        Forward pass for discriminator module of CCGAN.

        :param inputs: tuple containing the target property values and voxel grids
        :return: probability of input being fake or real
        """
        img, target = inputs
        target_output = self.label_condition_disc(target.view(-1, 1))
        target_output = target_output.view(-1, 1, 20, 20, 20)
        concat = torch.cat((img, target_output), dim=2)
        output = self.model(concat)
        reshape = output.view(img.size(0), output.size(2))
        linear = self.fc(reshape)

        return self.sig(linear)
