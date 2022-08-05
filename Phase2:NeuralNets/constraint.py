import torch
from torch.nn import Module, Embedding, Linear, Linear, LeakyReLU, Conv3d, BatchNorm3d, ReLU, Tanh, Sequential, Flatten, Dropout, Sigmoid

class Constraint(Module):
    def __init__(self):
        super(Constraint, self).__init__()

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
                   Linear(512, 1),
                   Linear(128, 1),
                   Linear(1,1))

    def forward(self, inputs):
        img, label = inputs
        # print(concat.size())
        output = self.model(inputs)
        return output