import torch
from torch.nn import Module, Embedding, Linear, Linear, LeakyReLU, Conv3d, BatchNorm3d, ReLU, Tanh, Sequential, Flatten, Dropout, Sigmoid

class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_condition_disc =
        Sequential(Linear(embedding_dim, 3 * 128 * 128))

        self.model =
        Sequential(Conv3d(in_channels=1, out_channels=4, kernel_size=(2,2,2),stride = 1, bias=False),
                   LeakyReLU(0.2, inplace=True),
                   Conv3d(in_channels=4, out_channels=8, kernel_size=(2,2,2),stride = 1, bias=False),
                   BatchNorm3d(4*8, momentum=0.1, eps=0.8),
                   LeakyReLU(0.2, inplace=True),
                   Conv3d(in_channels=8, out_channels=4, kernel_size=(2,2,2),stride = 1, bias=False),
                   BatchNorm3d(64 * 4, momentum=0.1, eps=0.8),
                   LeakyReLU(0.2, inplace=True),
                   Conv3d(in_channels=1, out_channels=4, kernel_size=(2,2,2),stride = 1, bias=False),
                   BatchNorm3d(64 * 8, momentum=0.1, eps=0.8),
                   LeakyReLU(0.2, inplace=True),
                   Flatten(),
                   Dropout(0.4),
                   Linear(4608, 1),
                   Sigmoid()
                      )

    def forward(self, inputs):
        img, label = inputs
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 3, 128, 128)
        concat = torch.cat((img, label_output), dim=1)
        # print(concat.size())
        output = self.model(concat)
        return output