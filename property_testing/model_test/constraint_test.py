from model.constraint import Constraint
import torch

dis = Constraint(grid_size=30)

image = torch.rand(1, 1, 30, 30, 30)

outputs = dis(image)

print(outputs)