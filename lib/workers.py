"""

workers

@author:    Matthew Holland

@email:     matthew.holland@my.utsa.edu

Date:       2026/04/15

    This file contains the objects and functions that provide worker capacity to certain objects in
the other libraries.

"""

#==================================================================================================
#
#   Imports
#
#==================================================================================================

import numpy as np
import scipy.interpolate as spint

#==================================================================================================
#
#   Interpolation workers
#
#==================================================================================================

def init_worker(cell_coords, data_matrix, points, neighbors, N_dims, N_chunk):
    global GLOBAL_cell_coords
    global GLOBAL_data_matrix
    global GLOBAL_points
    global GLOBAL_neighbors
    global GLOBAL_Ndims
    global GLOBAL_Nchunk

    GLOBAL_cell_coords = cell_coords
    GLOBAL_data_matrix = data_matrix
    GLOBAL_points = points
    GLOBAL_neighbors = neighbors
    GLOBAL_Ndims = N_dims
    GLOBAL_Nchunk = N_chunk


def rbf_interpolator_worker(j):
    import numpy as np
    import scipy.interpolate as spint

    coords = GLOBAL_cell_coords[j][:, :GLOBAL_Ndims]
    data   = GLOBAL_data_matrix[j]
    points = GLOBAL_points[j][:, :GLOBAL_Ndims]

    interp = spint.RBFInterpolator(
        coords,
        data.T,
        neighbors=GLOBAL_neighbors
    )

    n_pts  = points.shape[0]
    n_vars = data.shape[0]

    #result = np.memmap("temp_result.dat", dtype='float64', mode='w+', shape=(n_pts, n_vars) )
    result = np.zeros( (n_pts, n_vars) )

    chunk = GLOBAL_Nchunk

    for i in range(0, n_pts, chunk):
        pts_chunk = points[i:i+chunk]
        result[i:i+chunk] = interp(pts_chunk)

    return j, result

def lin_interpolator_worker(j):
    import numpy as np
    import scipy.interpolate as spint

    coords = GLOBAL_cell_coords[j][:, :GLOBAL_Ndims]
    data   = GLOBAL_data_matrix[j]
    points = GLOBAL_points[j][:, :GLOBAL_Ndims]

    interp = spint.LinearNDInterpolator(
        coords,
        data.T
    )

    n_pts  = points.shape[0]
    n_vars = data.shape[0]

    result = np.memmap(
                            "temp_result.dat",
                            dtype='float64',
                            mode='w+',
                            shape=(n_pts, n_vars)
                        )

    chunk = GLOBAL_Nchunk

    for i in range(0, n_pts, chunk):
        pts_chunk = points[i:i+chunk]
        result[i:i+chunk] = interp(pts_chunk)

    return j, result