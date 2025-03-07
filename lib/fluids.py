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

class generalFluid:
    """
        This class is dedicated towards producing data for a general fluid.

    """
    def __init__(self, data, t_points=0, nu=0, composition=None ):
        """
            Initialize the generalFluid object.

        
        """
        
        # Store fluid data
        self.nu = nu
        self.composition = composition

        # Store data dictionary
        self.data = data
        self.t_points=t_points

    def gradientField(cls, gradient_mesh, key=None, N_dims=3, coords=["x", "y", "z"] ):
        """
            Define the curl of the fluid field according to the input gradient field that 
        corresponds to a meshgrid point input

        Args:
            gradient_mesh (numpy ndarray - float):  The meshgrid-like input according to a meshgrid
                                                        of points:

                                                    (X, Y, Z)

        """

        # If no keys are given, just take them from the data dictionary
        if not key:
            key = list( cls.data.keys() )

        #
        for i in range(N_dims):
            print(f"For dimension {coords[i]}:")
            for k in key: 
                print(f"\tCreating key:\t d{k}/d{coords[i]}")
                cls.data[f"d{k}/d{coords[i]}"] = np.zeros_like( cls.data[k] )
                for j, t in enumerate( cls.t_points ):
                    print(f"\t\tSetting gradients at t={t:.3e} s.")
                    grad_data = np.gradient( cls.data[k][j,...], axis=i )[i]
                    print(f"Gradient of data shape:\t{np.shape(grad_data)}")
                    cls.data[f"d{k}/d{coords[i]}"][j,...] = grad_data / gradient_mesh[i] 

        # Store the keys where the gradients are
        cls.gradient_keys = key

        # Store the number of dimensions
        cls.gradient_N_dims = N_dims
        cls.gradient_coords = coords

    def curlField(cls, gradient_mesh, key=None, N_dims=None, coords=["x", "y", "z"], coord_loc=-1 ):
        """
            This method calculates the curl field based on which keys are vectors.

        Args:
            key (string, optional): The keys of the fields that will have the curl taken of them. 
                                        Defaults to None, which looks at which 
                                    

        """

        if not N_dims:
            N_dims = len(gradient_mesh)

        def filter_dict_by_key_prefix(d, k):
            # Convert k to lowercase for case-insensitive comparison
            k_lower = k.lower()
            # Create a new dictionary with keys that start with k
            filtered_dict = {key: value for key, value in d.items() if key.lower().startswith(k_lower)}
            return filtered_dict

        #
        for k in key: 
            print(f"Creating key:\tcurl({k})")
            cls.data[f"curl({k})"] = []
            for j, t in enumerate( cls.t_points ):
                print(f"\tAt time {t:.3e} s")

                # Set the vector field for the time point
                vector_field_dict = filter_dict_by_key_prefix( cls.data, k )
                vector_field = np.array( [vf for ky, vf in filter_dict_by_key_prefix( cls.data, k ).items()] )
                print(f"\t\tVector field: length {len(vector_field)}, where each is shape {np.shape(vector_field[0])}")
                axes = (1, 2, 3)
                if N_dims==2:
                    axes = (1, 2)
                grad_field = np.array( [np.gradient( v, axis=axes )[i] for i, v in enumerate( vector_field ) if i<N_dims] )#[:N_dims]
                print(f"\t\tGradient field is shape:\t{np.shape(grad_field)}")

                # Set the gradient field
                gradient_field = gradient_mesh
                print(f"\t\tAnd gradient field entries are shape {np.shape(gradient_field)}.")

                # Calculate the curl
                raw_curl = np.cross( 1/gradient_field, grad_field[:,j,...] , axis=0 )
                cls.data[f"curl({k})"] += [raw_curl]
                print(f"\t\tCurl shape is {np.shape(raw_curl)}")

            # Re-form the curl
            cls.data[f"curl({k})"] = np.asarray( cls.data[f"curl({k})"] )

        # Store the keys where the gradients are
        cls.curl_keys = key

        # Store the number of dimensions
        cls.curl_N_dims = N_dims
        cls.curl_coords = coords

    def sootFoil(cls, keys=None, t_bounds=None ):
        """
            This method simulates a soot foil simulation.

        Args:
            keys (string, optional):    The keys that will be integrated on to simulate the soot 
                                            foil. Default is None, which uses the ones stored in
                                            cls.curl_keys

            t_bounds (float, optional): The time bounds of integration. The default is None, which
                                            goes over the whole of the time steps.
        
        """

        # Define the keys
        if not keys:
            keys = cls.curl_keys
        else:
            raise ValueError("Does not automatically detect specified keys yet.")
        
        # Define the time bounds
        if not t_bounds:
            t_i_start = 0
            t_i_end = -1
        else:
            raise ValueError("Does not do bounds of integration yet.")
        
        # Do the integration
        for k in keys:
            cls.data[f"soot foil {k}"] = np.trapz( cls.data[f"curl({k})"], x=cls.t_points[t_i_start:t_i_end] )


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

        

        









