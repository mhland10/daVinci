"""

fluids

@author:    Matthew Holland
@email:     matthew.holland@my.utsa.edu
@date:      2025/02/17
@version:   0.0

    This module contains functions for fluid properties and calculations. This can span a variety
of flow regimes.

Version Date        Author              Changes

0.0     2025/02/17  Matthew Holland     Initial version of module

"""

#==================================================================================================
#
#   Imports
#
#==================================================================================================

import numpy as np
from transform import *

#==================================================================================================
#
#   Fluid Data Objects
#
#==================================================================================================

class compressibleGas:
    """
        This class contains the attributes and methods that pertain to a compressible gas as it
    pertains to data post-processing.

        The general idea is that the methods will collect data into the aptly named dictionary to 
    manipulate and analyze the data.

    """
    def __init__(self, dims=['t','x'] ):
        """
            Initialize the compressibleGas object. Only creates the data dictionary.

        args:
            N_dims (int, optional): The number of dimensions that the data will be in. Defaults to 
                                        2 since these will be the most prevalent cases.

        attributes:
            data:   dict    A dictionary to store the data for the object

        """
        
        self.data = {}

        self.dims = dims
        self.N_dims = len(dims)
        
        print("compressibleGas object created.")

    def shockTracking(cls, input_data , input_spatial_domain, input_time_domain, key="U:X", wt_family="bior1.3" , level=-1, coeff_index=0, store_wavelet=False ):
        """
            In this method, the presence of a shock will be tracked throughout time. The method
        uses the Discrete Wavelet Transform to track the discontinuity. 

        Args:
            input_data (dict):      The data to be analyzed. Stored as a dictionary. Data arrays
                                        must be stored in numpy arrays in the fomrat of:

                                        [t,x,y,z]

            input_spatial_domain (float):  The spatial domain of the data in the shape 3xN. The
                                            data must take the format of:

                                            [x,y,z]

            input_time_domain (float):  The time domain of the data. This is a 1D array of time.

            key (str, optional):    The key of the data that will be used to track the shock. 
                                        Defaults to "U:X".

            wt_family (str, optional):  The wavelet family that will find the shock. In general, it
                                            is recommended to stick the default, and if another is
                                            necessary, then one should select an odd wavelet that 
                                            pywavelets has access to. Defaults to "bior1.3". 
                                            Available wavelets are available at:

                                        https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html

            level (int, optional):  The level of the wavelet transform that the shock will be 
                                        tracked on. Defaults to -1.
                                        
        """

        # Check if the domain allows tracking a shock
        if cls.N_dims == 1:
            raise ValueError("The domain must have a time or second spatial axis to track a shock.")

        # Define the wavelet object that will be used to track the shock
        if store_wavelet:
            cls.shock_wavelet = WaveletData( input_data , N_dims=cls.N_dims, wavelet_family=wt_family )
            swt = cls.shock_wavelet
        else:
            shock_wavelet = WaveletData( input_data , N_dims=cls.N_dims, wavelet_family=wt_family )
            swt = shock_wavelet

        # Perform the wavelet transform on the data with the specified keys
        swt.waveletTransform([wt_family], keys=[key] )

        # Find the index of the shock location on a spatial domain that corresponds to the original 
        # data, but with the shape of the wavelet coefficients
        cls.shock_loc_indx = np.argmax( np.abs( swt.coeffs[wt_family][key][level][coeff_index] ) , axis=-1 )

        # Set up alternative domain
        cls.shock_loc = []
        if 'x' in cls.dims:
            print(f"Interpolating {input_data[key].shape[1]} points in [{input_spatial_domain[0][0]}, {input_spatial_domain[0][-1]}]")
            cls.x_pts = np.linspace(input_spatial_domain[0][0], input_spatial_domain[0][-1], swt.coeffs[wt_family][key][level][coeff_index].shape[-1] )
            cls.shock_loc += [cls.x_pts[cls.shock_loc_indx]]
        if 'y' in cls.dims:
            print(f"Interpolating {input_data[key].shape[1]} points in [{input_spatial_domain[0][0]}, {input_spatial_domain[0][-1]}]")
            cls.y_pts = np.linspace(input_spatial_domain[1][0], input_spatial_domain[1][-1], swt.coeffs[wt_family][key][level][coeff_index].shape[-1] )
            cls.shock_loc += [cls.y_pts[cls.shock_loc_indx]]
        if 'z' in cls.dims:
            cls.z_pts = np.linspace(input_spatial_domain[2][0], input_spatial_domain[2][-1], swt.coeffs[wt_family][key][level][coeff_index].shape[-1] )
            cls.shock_loc += [cls.z_pts[cls.shock_loc_indx]]
        if 't' in cls.dims:
            cls.t_pts = np.linspace(input_time_domain[0], input_time_domain[-1], swt.coeffs[wt_family][key][level][coeff_index].shape[0] )

        # Calculate the shock velocity
        if 't' in cls.dims:
            cls.shock_velocity = np.gradient( cls.shock_loc[0] , cls.t_pts , edge_order=2)


        return cls.shock_loc, cls.shock_velocity

        

        









