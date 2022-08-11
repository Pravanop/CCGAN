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

    def speciesToVoxel(self, structure, eden, image_type='Unified'):

        new_pos = self.basisTranslate(structure.cart_coords, self.dimension)

        atm_no_list = [atom.Z for atom in structure.species]
        species = np.unique(atm_no_list)  # list of unique atm no
        n_species = species.shape[0]  # number of unique atm

        new_pos_dict = {}

        for idx, atom in enumerate(atm_no_list):
            if atom not in new_pos_dict.keys():
                new_pos_dict[atom] = [new_pos[idx]]
            else:
                new_pos_dict[atom].append(new_pos[idx])

        density_matrix, species_matrix = self.iterator(new_pos_dict, eden)

        if image_type == 'Unified':
            image = np.zeros((self.dimension, self.dimension, self.dimension))
            species = np.zeros((self.dimension, self.dimension, self.dimension))
            for keys, values in density_matrix.items():
                image += values
            for keys, values in species_matrix.items():
                species += values

        elif image_type == 'Channels':
            image = np.zeros(n_species, self.dimension, self.dimension, self.dimension)
            species = np.zeros(n_species, self.dimension, self.dimension, self.dimension)
            i = 0
            for keys, values in density_matrix.items():
                image[i, :, :, :] = values
                i += 1
            i = 0
            for keys, values in species_matrix.items():
                species[i, :, :, :] = values
                i += 1

        return image, species

    def iterator(self, new_pos_dict, eden):

        density_matrix = {}
        species_matrix = {}
        for key, value in new_pos_dict.items():

            density_species = np.zeros((self.dimension, self.dimension, self.dimension))
            species_species = np.zeros((self.dimension, self.dimension, self.dimension))
            for pos in value:
                temp_density = np.zeros((self.dimension, self.dimension, self.dimension))
                temp_species = np.zeros((self.dimension, self.dimension, self.dimension))
                for idx, x in enumerate(temp_density):
                    for idy, y in enumerate(x):
                        for idz, z in enumerate(y):

                            r = (idx - pos[0]) ** 2 + (idy - pos[1]) ** 2 + (
                                    idz - pos[2]) ** 2  # square of euclidean distance between each voxel

                            temp_species[int(round(pos[0]))][int(round(pos[1]))][int(round(pos[2]))] = key

                            den = 2 * self.sigma ** 2  # denominator inside the exp.
                            if eden:  # electron density function
                                temp_density[idx][idy][idz] = 1.0 / ((2.0 * np.pi) ** 1.5) * key * (
                                        1.0 / self.sigma ** 3) * np.exp(
                                    -r / den)  # need to understand this better
                            else:
                                temp_density[idx][idy][idz] = key * np.exp(
                                    -r / den)
                density_species += temp_density
                species_species += temp_species
            density_matrix[key] = density_species
            species_matrix[key] = species_species

        return density_matrix, species_matrix
