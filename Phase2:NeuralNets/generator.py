import torch
from torch.nn import Module, Embedding, Linear, Linear, LeakyReLU, ConvTranspose3d, BatchNorm3d, ReLU, Tanh, Sequential
class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_conditioned_generator =
        Sequential(Linear(embedding_dim, 16))

        self.latent =
        Sequential(Linear(latent_dim, 4 * 4 * 512),
                      LeakyReLU(0.2, inplace=True))


        self.model =
        Sequential(ConvTranspose3d(in_channels= 1, out_channels=2, kernel_size=(1,1,1), stride = 1, bias = False),
                   BatchNorm3d(1*2, momentum=0.1, eps=0.8),
                   ReLU(True),
                   ConvTranspose3d(in_channels= 2, out_channels=4, kernel_size=(2,2,2), stride = 1, bias = False),
                   BatchNorm3d(2*4, momentum=0.1, eps=0.8),
                   ReLU(True),
                   ConvTranspose3d(in_channels= 4, out_channels=8, kernel_size=(2,2,2), stride = 1, bias = False),
                   BatchNorm3d(4*8, momentum=0.1, eps=0.8),
                   ReLU(True),
                   ConvTranspose3d(in_channels= 8, out_channels=16, kernel_size=(2,2,2), stride = 1, bias = False),
                   BatchNorm3d(8*8, momentum=0.1, eps=0.8),
                   ReLU(True),
                   ConvTranspose3d(in_channels= 8, out_channels=16, kernel_size=(3,3,3), stride = 1, bias = False),
                   Tanh())


    def forward(self, inputs):
        noise_vector, label = inputs
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512, 4, 4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)
        # print(image.size())
        return image