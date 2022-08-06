from model.constraint import Constraint
import torch

dis = Constraint(label_dim=1)

image = torch.rand(1, 1, 20, 20, 20)
property = torch.rand(1)
inputs = (image, property)

outputs = dis(inputs)

print(outputs)