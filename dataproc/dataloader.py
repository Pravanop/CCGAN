import torch
from torch.utils.data import Dataset
from dataproc.crystalToVoxel import cellToVoxel
from dataproc.MP_data import dataFromMp


class CrystalVoxelDataset(Dataset):

    def __init__(self,
                 pool,
                 property,
                 stability,
                 sigma,
                 grid_size):
        self.dataset = dataFromMp(pool=pool,
                                  property=property,
                                  stability=stability).dataset
        self.crysToVox = cellToVoxel(sigma=sigma,
                                     dimension=grid_size)

    def __len__(self):
        return int(self.dataset.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = f'{self.dataset[idx][1]}({self.dataset[idx][0]})'
        target = torch.tensor(self.dataset[idx][2], dtype=torch.float32)
        stability = torch.tensor(self.dataset[idx][3], dtype=torch.float32)
        structure = self.dataset[idx][4]

        voxel_np = self.crysToVox.speciesToVoxel(structure)  # ideally should be a transform property
        voxel = torch.tensor(voxel_np, dtype=torch.float32)
        sample = {'voxel': voxel, 'stability': stability, 'property': target, 'name': name}
        # name won't be used in training but is useful to have

        return sample
