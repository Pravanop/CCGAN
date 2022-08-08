import numpy as np


class cellToVoxel:

    def __init__(self,
                 sigma,
                 dimension):

        self.sigma = sigma
        self.dimension = dimension

    def basisTranslate(self, structure):

        pos = structure.cart_coords
        translate = self.dimension / 2
        cg = np.mean(pos, 0)
        basis = translate - cg
        npos = pos + basis
        return npos

    def speciesToVoxel(self, structure, eden=False):

        npos = self.basisTranslate(structure)
        voxel_all = np.zeros((self.dimension, self.dimension, self.dimension))
        for id, atom in enumerate(structure.species):
            voxel = np.zeros((self.dimension, self.dimension, self.dimension))
            for idx, x in enumerate(voxel):
                for idy, y in enumerate(x):
                    for idz, z in enumerate(y):
                        atm_no = atom.Z
                        r = (idx - npos[id][0]) ** 2 + (idy - npos[id][1]) ** 2 + (idz - npos[id][2]) ** 2
                        den = 2 * self.sigma ** 2
                        if eden:
                            voxel[idx][idy][idz] = 1.0 / ((2.0 * np.pi) ** 1.5) * atm_no * (1.0 / self.sigma ** 3) * np.exp(
                                -r / den)
                        else:
                            voxel[idx][idy][idz] = atm_no * np.exp(-r / den)
            voxel_all += voxel
        return voxel_all
