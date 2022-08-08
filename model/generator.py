from typing import Tuple

import torch
from torch.nn import Module, Linear, LeakyReLU, ConvTranspose3d, BatchNorm3d, ReLU, Tanh, Sequential


class Generator(Module):
    def __init__(self,
                 latent_dim: int = 40,
                 label_dim: int = 1) -> None:
        """
        Generator module hardcoded to output 20 dimensional cubic voxel grid.

        :param latent_dim: The input channels for embedding the noise vector
        :param label_dim: The dimension for property. Should mostly be 1.
        """
        super(Generator, self).__init__()

        self.latent = Sequential(Linear(latent_dim, 64),
                                 LeakyReLU(0.2, inplace=True))

        self.label_conditioned_generator = Linear(label_dim, 64)

        self.model = Sequential(
            ConvTranspose3d(in_channels=1, out_channels=2, kernel_size=(1, 2, 2), stride=2, bias=False,
                            output_padding=(1, 1, 1)),
            BatchNorm3d(2, momentum=0.1, eps=0.8),
            ReLU(True),
            ConvTranspose3d(in_channels=2, out_channels=4, kernel_size=(1, 2, 2), stride=1, bias=False),
            BatchNorm3d(4, momentum=0.1, eps=0.8),
            ReLU(True),
            ConvTranspose3d(in_channels=4, out_channels=8, kernel_size=(2, 3, 3), stride=1, bias=False),
            BatchNorm3d(8, momentum=0.1, eps=0.8),
            ReLU(True),
            ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=(2, 3, 3), stride=1, bias=False),
            BatchNorm3d(4, momentum=0.1, eps=0.8),
            ReLU(True),
            ConvTranspose3d(in_channels=4, out_channels=1, kernel_size=(2, 4, 4), stride=1, bias=False, dilation=2),
            Tanh())  # Tanh is a working choice

    def forward(self,
                inputs: Tuple[torch.tensor, torch.tensor] = None) -> torch.tensor:

        """

        :param inputs: tuple containing noise vectors and target property values
        :return: generated voxel images of crystal structures
        """
        noise_vector, target = inputs
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 1, 4, 4, 4)

        target_output = self.label_conditioned_generator(target.view(-1, 1))
        target_output = target_output.view(-1, 1, 4, 4, 4)

        concat = torch.cat((latent_output, target_output), dim=2)
        image = self.model(concat)

        return image
