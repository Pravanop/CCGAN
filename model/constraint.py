import torch
from torch.nn import Module, Linear, LeakyReLU, ReLU, Conv3d, BatchNorm3d, Sequential, Sigmoid, AdaptiveAvgPool3d


class Constraint(Module):
    def __init__(self) -> None:
        super(Constraint, self).__init__()

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
                                ReLU(),
                                AdaptiveAvgPool3d(output_size=(512, 1, 1))
                                )

    def forward(self,
                inputs: torch.tensor = None) -> torch.tensor:
        """
        Forward pass of constraint module.

        :param inputs: generated voxel images from generator
        :return: predictions for stability factor values
        """

        img = inputs
        output = self.model(img)
        reshape = output.view(img.size(0), output.size(2))
        linear = self.fc(reshape)

        return linear
