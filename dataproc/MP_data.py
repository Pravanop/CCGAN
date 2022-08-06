from typing import List
import pymatgen as mp
from mp_api import MPRester
import numpy as np

class dataFromMp:

    def __init__(self,
                 pool: List[str],
                 property: str,
                 stability: str):

        self.api_key = 'u1TjwfwfTnpF8IolXF9PBY9RT9YauL84' #pravan's api key, change if needed
        self.pool = pool

        self.property = property
        self.stability = stability

        with MPRester(self.api_key) as mpr:
            self.docs = mpr.summary.search(elements=self.pool,
                                      fields=[self.property,
                                              self.stability,
                                              "formula_pretty",
                                              "material_id"]) #formula and material_id needed for easy pre-processing

            self.structs = mpr.materials.search(elements=self.pool,
                                        fields=["initial_structures",
                                                "material_id"])

        self.docs = [dict(ele) for ele in self.docs]
        self.structs = [dict(ele) for ele in self.structs] #intial structures

        self.dataset = self.ordering()


    @staticmethod
    def search(arr, n, x):
        #TODO using linear search, change to more efficient method later
        for i in range(0, n):
            if (arr[i]['material_id'] == x):
                return i


    def ordering(self):
        data_int = []
        dtype = [('ID', 'U40'), ('Formula', 'U40'), ('Property', np.float64), ('Stability', np.float64), ('Structure', mp.core.Structure)]
        for idx, doc in enumerate(self.docs):

            idx_struct = self.search(self.structs, len(self.docs), doc['material_id'])

            if (doc['material_id'] == self.structs[idx_struct]['material_id']):
               structure = self.structs[idx_struct]['initial_structures'][0]
               row_temp = np.array((str(self.structs[idx_struct]['material_id']),
                                    str(doc['formula_pretty']),
                                    doc[self.property],
                                    doc[self.stability],
                                    structure), dtype=dtype )

            data_int.append(row_temp)

        return np.array(data_int)


