from model.generator import Generator
import torch
gen = Generator(latent_dim=32,
                label_dim=1)

noise = torch.rand(32)
property = torch.rand(1)
inputs = (noise, property)

outputs = gen(inputs)

print(outputs.size())