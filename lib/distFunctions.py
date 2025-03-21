"""

**distFunctions.py**

@author:    Matthew Holland
@contact:   matthew.holland@my.utsa.edu

    This file contains helpful functions for the other libraries

"""

#==================================================================================================
#
#   Import modules
#
#==================================================================================================

from numba import prange, njit
import numpy as np

#==================================================================================================
#
#   Vector field functions
#
#==================================================================================================

@njit
def timeStep_crossProducts( gradient_mesh, t_points, data_dict, key ):
    """
        Calculate the cross product of the gradient field and the vector field at each time point

    Args:
        gradient_mesh (float - numpy ndarray): The mesh of the gradient field
        t_points (_type_): _description_
        data_dict (_type_): _description_
        key (_type_): _description_

    Returns:
        _type_: _description_
    """


    # Initialize the curl product array
    curl_product = np.zeros( ( len(t_points) ,) + np.shape(gradient_mesh)[1:] )

    # Get the vector field as an array from the data dictionary
    vector_field = np.array( [vf for ky, vf in data_dict.items()] )

    # Get the gradient field for the vector field
    axes = (1, 2, 3)
    if len(gradient_mesh)==2:
        axes = (1, 2)
    grad_field = np.array( [np.gradient( v, axis=axes )[i] for i, v in enumerate( vector_field ) if i<len(gradient_mesh)] )#[:N_dims]


    for j in prange(len(t_points)):
        t = t_points[j]

        # Calculate the curl
        curl_product[j] = np.cross( 1/gradient_mesh, grad_field[:,j,...] , axis=0 )

    return curl_product

#==================================================================================================
#
#   Mathematical functions
#
#==================================================================================================

#@njit
def filteredAverage_xDict( data_dict, averaging_axis=0, filter_nan=True, filter_num=None ):    
    """
        This function takes a dictionary of arrays/matrices and averages them, but filters out NaN
    values and removes the entry from the data to make a filtered average from the data.

    Args:
        data_dict (dictionary): A dictionary of numpy arrays or matrices that contains the data.

        averaging_axis (int, optional):     The axis to take the averaging over. Defaults to 0.

    """

    # Find the original data shape and keys
    og_data_shape = np.shape( list(data_dict.values())[0] )
    data_keys = list( data_dict.keys() )


    # Initialize the filtered average dictionary
    filtered_avg_dict = {}
    for i in prange( len( data_keys) ):

        # Pull the data array, then move the axis if necessary, and flatten the back end
        og_data_array = data_dict[ data_keys[i] ]
        if not averaging_axis==0:
            data_array = np.moveaxis( og_data_array, averaging_axis, 0 )
        else:
            data_array = og_data_array
        og_shape = np.shape( data_array )
        data_array_flat = np.reshape( data_array, ( og_shape[0], -1 ) )

        # Find the number of points
        N_pts = np.prod( np.shape(data_array)[1:] )

        # Initialize the filtered average array
        data_array_avg = np.zeros( N_pts )

        # Loop through the data array and average the data
        for j in prange( N_pts ):
            dats = []
            data_array = data_array_flat[:,j]
            for k in prange( len(data_array) ):
                if not np.isnan( data_array[k] ) and filter_nan:
                    if not data_array[k]==filter_num:
                        dats.append( data_array[k] )
            data_array_avg[j] = np.mean( np.array( dats ) )

        # Reshape the data array
        data_array_ent = np.reshape( data_array_avg, og_shape[1:] )

        # Move the axis back if necessary
        if not averaging_axis==0:
            data_array_ent = np.moveaxis( data_array_ent, 0, averaging_axis )

        # Add the data array to the filtered average dictionary
        filtered_avg_dict[ data_keys[i] ] = data_array_ent

    return filtered_avg_dict

            



    



