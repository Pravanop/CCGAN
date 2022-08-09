from typing import Union
import plotly.graph_objects as go
import numpy as np


def plotter_voxel(sample: Union[dict, np.array] = None,
                  type: str = 'density') -> None:
    """
    The main function to plot voxel grids as Plotly Interactive Plots

    :param sample: Either a dictionary sample from
    the Pytorch Dataset or a voxel grid numpy array
    :param type: Specifying if the voxel to be printed should be a density or species matrix.
    Takes in 'density' or 'species'.
    :return: An interactive 3D plot will be opened on a local port.
    """
    if isinstance(sample, dict):
        if type == 'density':
            voxel = sample['voxel'].numpy()

        else:
            voxel = sample['species matrix'].numpy()

        dimension = voxel.shape[0]
        X, Y, Z = np.mgrid[0:dimension:20j, 0:dimension:20j, 0:dimension:20j]
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=voxel.flatten(),
            isomin=0,
            isomax=np.max(voxel),
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=40,  # needs to be a large number for good volume rendering
        ))
        fig.update_layout(title_text=sample['name'])
        fig.show()

    else:
        dimension = sample.shape[0]
        X, Y, Z = np.mgrid[0:dimension:20j, 0:dimension:20j, 0:dimension:20j]
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=sample.flatten(),
            isomin=0,
            isomax=np.max(sample),
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=40,  # needs to be a large number for good volume rendering
        ))
        fig.update_layout(title_text='Reconstructed gaussian filter')
        fig.show()
