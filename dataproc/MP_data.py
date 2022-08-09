from typing import List, Union
import pymatgen as mp
from mp_api import MPRester
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class dataFromMp:

    def __init__(self,
                 pool: Union[List[str], str] = None,
                 target: str = None,
                 stability: str = None) -> None:

        """
        Script to use Materials Project APIs to extract material information as ready-made dataset.

        :param pool: Either a list of elements that will result in search for compounds containing these elements or
        a formula of the form **O3 that searches for all compounds of the form ABO3 :param target: The property of
        the material that you are basing the inverse design for
        :param stability: The property of the material that will determine the stability of the material generated by
        the network

        """

        self.api_key = 'u1TjwfwfTnpF8IolXF9PBY9RT9YauL84'  # pravan's api key, change if needed
        self.pool = pool

        self.property = target
        self.stability = stability

        with MPRester(self.api_key) as mpr:
            if isinstance(pool, list):
                self.docs = mpr.summary.search(elements=self.pool,
                                               fields=[self.property,
                                                       self.stability,
                                                       "formula_pretty",
                                                       "material_id"])  # formula and material_id needed for easy pre-processing

                self.docs = [dict(ele) for ele in self.docs]
                mp_ids = [self.docs[i]['material_id'] for i in range(len(self.docs))]
                self.structs = mpr.materials.search(task_ids=mp_ids,
                                                    fields=["initial_structures",
                                                            "material_id"])  # need a seperate call for getting structures

            else:

                self.docs = mpr.summary.search(formula=self.pool,
                                               fields=[self.property,
                                                       self.stability,
                                                       "formula_pretty",
                                                       "material_id"])

                self.docs = [dict(ele) for ele in self.docs]
                mp_ids = [self.docs[i]['material_id'] for i in range(len(self.docs))]
                self.structs = mpr.materials.search(task_ids=mp_ids,
                                                    fields=["initial_structures",
                                                            "material_id"])

        self.structs = [dict(ele) for ele in self.structs]  # converting to dict for ease of coding up next steps

    @staticmethod
    def search(arr: np.array = None,
               n: int = None,
               x: Union[float, int, str] = None) -> int:
        """
        Linear search algorithm to search for a material id in the different dictionaries.

        :param arr: The array to search in
        :param n: The length of array
        :param x: The element to search for
        :return: index of the element in the array
        """

        for i in range(0, n):
            if arr[i]['material_id'] == x:
                return i

    @property
    def crystalInfo(self) -> np.array:
        """
        The primary function that orders the extracted data into a Structured numpy array. Every sample will have a
        materials project assigned id, a formula, a target property, a stability factor and a crystal pymatgen
        Structure.
        :return: numpy array with all the samples called from materials dataset ordered as required
        """
        data_int = []
        datatype = [('ID', 'U40'), ('Formula', 'U40'), ('Property', np.float64), ('Stability', np.float64),
                    ('Structure', mp.core.Structure)]
        # !!!!!!!!!!!!! This is going to be a standard format for this project !!!!!!!!!!!!!!
        for idx, doc in enumerate(self.docs):

            idx_struct = self.search(self.structs, len(self.docs),
                                     doc['material_id'])  # both dictionaries do not line up

            if doc['material_id'] == self.structs[idx_struct]['material_id']:
                structure = self.structs[idx_struct]['initial_structures'][0]
                row_temp = np.array((str(self.structs[idx_struct]['material_id']),
                                     str(doc['formula_pretty']),
                                     doc[self.property],
                                     doc[self.stability],
                                     structure), dtype=datatype)

                data_int.append(row_temp)
            else:
                print('Something is wrong with the data extracted. The material ids are not matching up. Could lead '
                      'to serious confusions!')

        return np.array(data_int)
