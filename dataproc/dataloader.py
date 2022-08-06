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

        name = self.dataset[idx][1]
        property = self.dataset[idx][2]
        stability = self.dataset[idx][3]
        structure = self.dataset[idx][4]

        voxel = self.crysToVox.speciesToVoxel(structure) #ideally should be a transform property

        sample = {'voxel': voxel, 'stability': stability, 'property': property, 'name': name}
        #name won't be used in training but is useful to have

        return sample
