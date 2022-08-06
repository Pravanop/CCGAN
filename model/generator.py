import torch
from torch.nn import Module, Linear, LeakyReLU, ConvTranspose3d, BatchNorm3d, ReLU, Tanh, Sequential


class Generator(Module):
    def __init__(self,
                 latent_dim: int = 40,
                 label_dim: int = 1):
        super(Generator, self).__init__()

        self.latent = Sequential(Linear(latent_dim, 64),
                                 LeakyReLU(0.2, inplace=True))

        self.label_conditioned_generator = Linear(label_dim, 64)

        self.model = Sequential(
            ConvTranspose3d(in_channels=1, out_channels=2, kernel_size=(1, 2, 2), stride=2, bias=False, output_padding=(1, 1, 1)),
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
            Tanh())

    def forward(self, inputs):
        noise_vector, label = inputs
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 1, 4, 4, 4)

        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4, 4)

        concat = torch.cat((latent_output, label_output), dim=2)
        print(concat.size())
        image = self.model(concat)

        return image
