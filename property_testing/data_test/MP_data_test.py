from dataproc.dataloader import CrystalVoxelDataset as CVD
from torch.utils.data import DataLoader

dataset = CVD(pool=['Ba', 'Ti', 'O'],
              property = 'band_gap',
              stability= 'energy_above_hull',
              sigma=0.3,
              grid_size=20)
dataloader = DataLoader(dataset, batch_size=16,
                        shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['voxel'].size(),
          sample_batched['property'].size(), sample_batched['stability'].size())