from typing import Union
import numpy as np
import pymatgen.core


class cellToVoxel:

    def __init__(self,
                 sigma: Union[int, float] = None,
                 dimension: int = None) -> None:
        """
        Convert a crystal structure to a cubic voxel grid of electron density or nuclear potentiol.

        :param sigma: The standard deviation for the density gaussian function
        :param dimension: The dimensions of the cubic voxel grid
        """

        self.sigma = sigma
        self.dimension = dimension

    def basisTranslate(self,
                       pos: list = None) -> list:
        """
        Translates the positions of the atoms such that the cell is centred at the centre of the voxel grid.

        :param pos: List of cartesian coordinates of all atoms in the Structure
        :return: list of new positions
        """

        translate = self.dimension / 2
        cg = np.mean(pos, 0)
        basis = translate - cg
        new_pos = pos + basis
        return new_pos

    def speciesToVoxel(self,
                       structure: pymatgen.core.Structure = None,
                       eden: bool = None) -> np.array:
        """
        The primary function to create the electron density voxel grid or simple nuclear potential voxel grid.

        :param structure: PyMatGen Structure module
        :param eden: boolean for 'e'lectronic 'den'sity or nuclear potential model
        :return: 3-D cubic voxel grid of specified dimension and type
        """

        new_pos = self.basisTranslate(structure.cart_coords)
        voxel_all = np.zeros((self.dimension, self.dimension, self.dimension))
        for id_atom, atom in enumerate(structure.species):  # iterating over atoms
            voxel = np.zeros((self.dimension, self.dimension, self.dimension))
            for idx, x in enumerate(voxel):
                for idy, y in enumerate(x):
                    for idz, z in enumerate(y):
                        atm_no = atom.Z
                        r = (idx - new_pos[id_atom][0]) ** 2 + (idy - new_pos[id_atom][1]) ** 2 + (
                                idz - new_pos[id_atom][2]) ** 2  # square of euclidean distance between each voxel
                        # and the atom
                        den = 2 * self.sigma ** 2  # denominator inside the exp.
                        if eden:  # electron density function
                            voxel[idx][idy][idz] = 1.0 / ((2.0 * np.pi) ** 1.5) * atm_no * (
                                    1.0 / self.sigma ** 3) * np.exp(
                                -r / den)  # need to understand this better
                        else:
                            voxel[idx][idy][idz] = atm_no * np.exp(
                                -r / den)  # simple nuclear potential gaussian function
            voxel_all += voxel
        return voxel_all
