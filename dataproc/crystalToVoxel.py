from typing import Union, Tuple
import numpy as np
import pymatgen.core


class cellToVoxel:

    def __init__(self,
                 sigma: Union[int, float] = None,
                 dimension: int = None) -> None:
        """
        Convert a crystal structure to a cubic voxel grid of electron density or nuclear potential.

        :param sigma: The standard deviation for the density gaussian function
        :param dimension: The dimensions of the cubic voxel grid
        """

        self.sigma = sigma
        self.dimension = dimension

    @staticmethod
    def basisTranslate(pos: list = None,
                       dimension: int = 20) -> list:
        """
        Translates the positions of the atoms such that the cell is centred at the centre of the voxel grid.

        :param dimension: size of voxel grid
        :param pos: List of cartesian coordinates of all atoms in the Structure
        :return: list of new positions
        """

        translate = dimension / 2
        cg = np.mean(pos, 0)
        basis = translate - cg
        new_pos = pos + basis

        return new_pos

    def speciesToVoxel(self,
                       structure: pymatgen.core.Structure = None,
                       eden: bool = None) -> Tuple[np.array, np.array]:
        """
        The primary function to create the electron density voxel grid or simple nuclear potential voxel grid.

        :param structure: PyMatGen Structure module
        :param eden: boolean for 'e'lectronic 'den'sity or nuclear potential model
        :return: 3-D cubic voxel grid of specified dimension and type
        """

        new_pos = self.basisTranslate(structure.cart_coords, self.dimension)
        voxel_all = np.zeros((self.dimension, self.dimension, self.dimension))
        species_all = np.zeros((self.dimension, self.dimension, self.dimension))

        for id_atom, atom in enumerate(structure.species):  # iterating over atoms
            voxel = np.zeros((self.dimension, self.dimension, self.dimension))
            species = np.zeros((self.dimension, self.dimension, self.dimension))
            for idx, x in enumerate(voxel):
                for idy, y in enumerate(x):
                    for idz, z in enumerate(y):
                        atm_no = atom.Z

                        r = (idx - new_pos[id_atom][0]) ** 2 + (idy - new_pos[id_atom][1]) ** 2 + (
                                idz - new_pos[id_atom][2]) ** 2  # square of euclidean distance between each voxel
                        # and the atom
                        if r ** 0.5 < 0.5:  # needs altering
                            if species[idx][idy][idz] > 0.0:
                                if np.random.rand() > 0.5:
                                    species[idx][idy][idz] = atm_no
                            else:
                                species[idx][idy][idz] = atm_no

                        den = 2 * self.sigma ** 2  # denominator inside the exp.
                        if eden:  # electron density function
                            voxel[idx][idy][idz] = 1.0 / ((2.0 * np.pi) ** 1.5) * atm_no * (
                                    1.0 / self.sigma ** 3) * np.exp(
                                -r / den)  # need to understand this better
                        else:
                            voxel[idx][idy][idz] = atm_no * np.exp(
                                -r / den)  # simple nuclear potential gaussian function
            voxel_all += voxel
            species_all += species
        return voxel_all, species_all

    def speciesToVoxel2(self,
                        structure: pymatgen.core.Structure = None,
                        eden: bool = None,
                        image_type: str = 'Unified') -> Union[Tuple[np.array, np.array], Tuple[list, list]]:
        """
        The primary function to create the electron density voxel grid or simple nuclear potential voxel grid.

        :param image_type: Type of density matrix shape
        :param structure: PyMatGen Structure module
        :param eden: boolean for 'e'lectronic 'den'sity or nuclear potential model
        :return: 3-D cubic voxel grid of specified dimension and type
        """

        new_pos = self.basisTranslate(structure.cart_coords, self.dimension)
        atm_no_list = [atom.Z for atom in structure.species]
        species = np.unique(atm_no_list)
        n_species = species.shape[0]

        if 'Unified' == image_type:
            image = np.zeros((self.dimension, self.dimension, self.dimension))
            species_mat = np.zeros((self.dimension, self.dimension, self.dimension))

            image, species_mat = self.iterator(image, species_mat, atm_no_list, species, new_pos, eden)

            image_unified = np.zeros((self.dimension, self.dimension, self.dimension))
            species_unified = np.zeros((self.dimension, self.dimension, self.dimension))

            for i in range(n_species):
                image_unified += image[i]
                species_unified += species_mat[i]

            return image_unified, species_unified

        elif image_type == 'Separate':
            image = [np.zeros(self.dimension, self.dimension, self.dimension) for __ in range(n_species)]
            species_mat = [np.zeros(self.dimension, self.dimension, self.dimension) for __ in range(n_species)]

            image, species_mat = self.iterator(image, species_mat, atm_no_list, species, new_pos, eden)

            return image, species_mat

        elif image_type == 'Channels':
            image = [np.zeros(self.dimension, self.dimension, self.dimension) for __ in range(n_species)]
            species_mat = [np.zeros(self.dimension, self.dimension, self.dimension) for __ in range(n_species)]

            image = self.iterator(image, species_mat, atm_no_list, species, new_pos, eden)
            image_channel = np.zeros(n_species, self.dimension, self.dimension, self.dimension)
            species_mat_channel = np.zeros(n_species, self.dimension, self.dimension, self.dimension)
            for i in range(n_species):
                image_channel[i, :, :, :] = image[i]
                species_mat_channel[i, :, :, :] = species_mat[i]

            return image_channel, species_mat_channel

    def iterator(self, image, species_mat, atm_no_list, species, new_pos, eden):

        dimension = image[0].shape[-1]

        for idx in range(dimension):
            for idy in range(dimension):
                for idz in range(dimension):

                    for id_atom, atom in atm_no_list:
                        channel_no = np.where(species == atom)

                        r = ((idx - new_pos[id_atom][0]) ** 2 + (idy - new_pos[id_atom][1]) ** 2 + (
                                idz - new_pos[id_atom][2]) ** 2) ** 0.5

                        # for density matrix
                        if eden:
                            image[channel_no][idx][idy][idz] = 1.0 / ((2.0 * np.pi) ** 1.5) * atom * (
                                    1.0 / self.sigma ** 3) * np.exp(
                                -r ** 2 / (2 * (self.sigma ** 2)))
                        else:
                            image[channel_no][idx][idy][idz] = atom * np.exp(
                                -r ** 2 / (2 * (self.sigma ** 2)))

                        # for species matrix
                        if r < 0.5:
                            if species_mat[channel_no][idx][idy][idz] > 0.0:
                                if np.random.rand() > 0.5:
                                    species_mat[channel_no][idx][idy][idz] = atom
                            else:
                                species_mat[channel_no][idx][idy][idz] = atom

        return image, species_mat
