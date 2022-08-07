

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['voxel'].size(),
          sample_batched['property'].size(), sample_batched['stability'].size())