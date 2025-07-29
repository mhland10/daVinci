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
from sciDataRead import sweep
from distFunctions import *

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
            cls.data[f"curl({k})"] = np.zeros( ( len(cls.t_points) ,) + np.shape(gradient_mesh)[1:] )

            # Set the vector field for all time points
            vector_field = np.array( [vf for ky, vf in filter_dict_by_key_prefix( cls.data, k ).items()] )
            print(f"\t\tVector field: length {len(vector_field)}, where each is shape {np.shape(vector_field[0])}")
            axes = (1, 2, 3)
            if N_dims==2:
                axes = (1, 2)
            """
            for i, v in enumerate( vector_field ):
                if i < N_dims:
                    print(f"i:\t{i}")
                    print(f"axes:\t{axes}")
                    print(f"v:\t{v}")
                    print(f"Gradient shape:\t{np.shape(np.gradient( v, axis=np.array(axes)-1 ))}")
                    print(f"Gradient sample:\t{np.gradient( v, axis=np.array(axes)-1 )}")
            """
            grad_field = np.array( [np.gradient( v, axis=np.array(axes)-1 )[i] for i, v in enumerate( vector_field ) if i<N_dims] )
            print(f"Gradient field is shape:\t{np.shape(grad_field)}")
            print(f"And gradient field entries are shape {np.shape(gradient_mesh)}.")

            for j, t in enumerate( cls.t_points ):
                print(f"\tAt time {t:.3e} s")

                # Calculate the curl
                raw_curl = np.cross( 1/gradient_mesh, grad_field[:,j,...] , axis=0 )
                cls.data[f"curl({k})"][j] = raw_curl
                print(f"\t\tCurl shape is {np.shape(raw_curl)}")

        # Store the keys where the gradients are
        cls.curl_keys = key

        # Store the number of dimensions
        cls.curl_N_dims = N_dims
        cls.curl_coords = coords

    def sootFoil(cls, keys=None, t_bounds=None, integration_axis=0 ):
        """
            This method simulates a soot foil simulation.

        Args:
            keys (string, optional):    The keys that will be integrated on to simulate the soot 
                                            foil. Default is None, which uses the ones stored in
                                            cls.curl_keys

            t_bounds (float, optional): The time bounds of integration. The default is None, which
                                            goes over the whole of the time steps. Must be in the 
                                            format:

                                            [ start time, end time ]
        
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
            start_errors = np.abs( t_bounds[0]/cls.t_points - 1 )
            t_i_start = np.argmin( start_errors )
            end_errors = np.abs( t_bounds[1]/cls.t_points - 1 )
            t_i_end = np.argmin( end_errors )
        
        # Do the integration
        for k in keys:
            cls.data[f"soot foil {k}"] = np.trapz( cls.data[f"curl({k})"][t_i_start:t_i_end], x=cls.t_points[t_i_start:t_i_end], axis=integration_axis )


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

    def shockTracking(cls, input_data , input_spatial_domain, input_time_domain, key="U:X", wt_family="bior1.3", 
                      level=-1, coeff_index=0, store_wavelet=False, nonuniform_dims=[" "], 
                      filter_nan=True, gradient_order=None, window_lims=2 ):
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
        if 't' in cls.dims:
            swt.N_dims -= 1
        swt.waveletTransform([wt_family], keys=[key] )

        # Find the domain that represents the DWT
        used_domain = []
        if 't' in cls.dims:
            used_domain += [input_time_domain]
            swt.N_dims += 1
        if 'x' in cls.dims:
            used_domain += [input_spatial_domain[0]]
        if 'y' in cls.dims: 
            used_domain += [input_spatial_domain[1]]
        if 'z' in cls.dims:
            used_domain += [input_spatial_domain[2]]
        cls.used_domain = used_domain
        swt.domains( used_domain )

        # Find the index of the shock location on a spatial domain that corresponds to the original 
        # data, but with the shape of the wavelet coefficients
        if filter_nan:
            cls.shock_loc_indx = np.nanargmax( np.abs( np.array(swt.coeffs[wt_family][key][level])[:,window_lims:-window_lims] ) , axis=-1 )+window_lims
        else:
            cls.shock_loc_indx = np.argmax( np.abs( np.array(swt.coeffs[wt_family][key][level])[:,window_lims:-window_lims] ) , axis=-1 )+window_lims

        # Set up alternative domain
        cls.shock_loc = []
        if 'x' in cls.dims:
            print(f"x data is in {cls.dims.index('x')}")
            cls.x_pts = swt.domain[level][cls.dims.index('x')]
            cls.shock_loc += [cls.x_pts[cls.shock_loc_indx]]
        if 'y' in cls.dims:
            cls.y_pts = swt.domain[cls.dims.index('y')][level]
            cls.shock_loc += [cls.y_pts[cls.shock_loc_indx]]
        if 'z' in cls.dims:
            cls.z_pts = swt.domain[level][cls.dims.index('z')]
            cls.shock_loc += [cls.z_pts[cls.shock_loc_indx]]
        if 't' in cls.dims:
            cls.t_pts = input_time_domain

        # Calculate the shock velocity
        if 't' in cls.dims:
            if not gradient_order or gradient_order==1:
                cls.shock_velocity = np.gradient( cls.shock_loc[0] , cls.t_pts , edge_order=2)
            elif gradient_order>1:
                num_grad = numericalGradient( 1, ( gradient_order//2, gradient_order//2 ) )
                cls.shock_velocity = num_grad.gradientCalc( cls.t_pts, cls.shock_loc[0] )
                cls.num_grad = num_grad


        # Store the domain for later
        cls.og_spatial_domain = input_spatial_domain
        cls.og_time_domain = input_time_domain

        return cls.shock_loc, cls.shock_velocity
    
    def frozenShockProfile(cls, reader_dir, t_lims=None, step_mults=(1.0, 1e3), N_sweep=1000, data_file_lead="post00*", file_format="h5", reader_dims=['x','y'], reader_interpolator="lin" ):
        """
            This method will track the profile along a shock that is tracked via the 
        shockTracking() method

        Args:

        """
        print("Under Construction")

        # Interpolate the shock location to the original coordinates
        if not t_lims:
            cls.og_shock_loc = np.interp( cls.og_time_domain, cls.t_pts, cls.shock_loc[0] )
        else:
            # Filter time points within bounds
            og_times_filt = [time for time in cls.og_time_domain if np.min(t_lims) <= time <= np.max(t_lims[1])]
            cls.og_shock_loc = np.interp( og_times_filt , cls.t_pts, cls.shock_loc[0] )

        # Define the anchor of the sweep
        if 'x' in cls.dims:
            cls.anchors = np.array([ cls.og_shock_loc, np.zeros_like(cls.og_shock_loc), np.zeros_like(cls.og_shock_loc) ])
        elif 'y' in cls.dims:  
            cls.anchors = np.array([ np.zeros_like(cls.og_shock_loc), cls.og_shock_loc, np.zeros_like(cls.og_shock_loc) ])
        elif 'z' in cls.dims:
            cls.anchors = np.array([ np.zeros_like(cls.og_shock_loc), np.zeros_like(cls.og_shock_loc), cls.og_shock_loc ])
        else:
            raise ValueError("No spatial dimensions to sweep.")

        # Define the delta of the sweep
        gradients = np.array( [ np.array( np.gradient(cls.og_spatial_domain) )[i][i] for i in range(cls.N_dims) ] )
        base_step = np.min( np.linalg.norm( gradients, axis=0 ) )
        min_step = np.min( step_mults ) * base_step
        max_step = np.max( step_mults ) * base_step 
        deltas_array = np.logspace( np.log10(min_step), np.log10(max_step), num=N_sweep//2 )
        if 'x' in cls.dims:
            cls.deltas = np.array([ deltas_array, np.zeros_like(deltas_array), np.zeros_like(deltas_array) ])
        elif 'y' in cls.dims:
            cls.deltas = np.array([ np.zeros_like(deltas_array), deltas_array, np.zeros_like(deltas_array) ])
        elif 'z' in cls.dims:
            cls.deltas = np.array([ np.zeros_like(deltas_array), np.zeros_like(deltas_array), deltas_array ])
        cls.base_step = base_step
        cls.gradients = gradients
        
        # Define the sweep and read from the files
        frozen_shock = sweep( cls.anchors, np.concatenate(( -cls.deltas[:,::-1], cls.deltas ), axis=-1), data_file_lead, file_format=file_format, t_lims=t_lims )
        frozen_shock.hdf5DataRead( reader_dir, interpolator=reader_interpolator.lower(), dims=reader_dims )
        cls.frozen_shock = frozen_shock

        # Provide a filtered version of the data
        cls.frozen_shock_profile = {}
        for key, data in cls.frozen_shock.data.items():
            cls.frozen_shock_profile[key] = np.nanmean( data, axis=0 )


#============================

        

        









