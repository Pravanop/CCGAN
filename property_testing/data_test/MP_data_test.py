from dataproc.MP_data import dataFromMp
from dataproc.crystalToVoxel import cellToVoxel

dataset = dataFromMp(pool='Ba*O3',
                     stability='energy_above_hull',
                     target= 'band_gap',
                     ).crystalInfo

structure = dataset[100][-1]
image, species = cellToVoxel(sigma=0.3,
                             dimension=20).speciesToVoxel2(structure, eden=False)


