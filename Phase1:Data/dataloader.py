import torch
import MP_data
from torch.utils.data import Dataset
from crystalToVoxel import cellToVoxel


class CrystalVoxelDataset(Dataset):

    def __init__(self,
                 pool,
                 property,
                 stability,
                 sigma,
                 grid_size):
        self.dataset = MP_data.dataFromMp(pool=pool,
                                          property=property,
                                          stability=stability).dataset
        self.crysToVox = cellToVoxel(sigma=sigma,
                                     dimension=grid_size)

    def __len__(self):
        return int(self.dataset.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        stability = self.dataset[idx][3]
        property = self.dataset[idx][2]
        name = self.dataset[idx][1]
        structure = self.dataset[idx][3]
        voxel = self.crysToVox.speciesToVoxel(structure)
        sample = {'voxel': voxel, 'stability': stability, 'property': property, 'name': name}

        return sample
