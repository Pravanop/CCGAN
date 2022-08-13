from dataproc.MP_data import dataFromMp
from dataproc.crystalToVoxel import cellToVoxel
from utils.visualization import plotter_voxel

dataset = dataFromMp(pool='**O3',
                     stability='energy_above_hull',
                     target='band_gap',
                     ).crystalInfo

structure = dataset[0][-1]
print(dataset[0][0])
image, species = cellToVoxel(sigma=0.3,
                             dimension=20).speciesToVoxel(structure, eden=False)

plotter_voxel(image)
