import torch
from torch.nn import Module, Linear, LeakyReLU, Conv3d, BatchNorm3d, Sequential, Sigmoid, AdaptiveAvgPool3d


class Discriminator(Module):
    def __init__(self,
                 label_dim: int = 1):
        super(Discriminator, self).__init__()

        self.label_condition_disc = Linear(label_dim, 20 * 20 * 20)
        self.fc2 = Linear(512, 1)
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
                                )

    def forward(self, inputs):

        img, label = inputs
        label_output = self.label_condition_disc(label.view(-1, 1))
        label_output = label_output.view(-1, 1,  20, 20, 20)
        concat = torch.cat((img, label_output), dim=2)
        output = self.model(concat)
        reshape = output.view(img.size(0), output.size(2))
        linear = self.fc2(reshape)

        return self.sig(linear)
