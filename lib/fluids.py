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
            swt.max_levels = pywt.dwtn_max_level( ( input_data[list(input_data.keys())[0]].shape[-1] ,), wt_family )
        swt.waveletTransform([wt_family], keys=[key] )

        # Find the domain that represents the DWT
        used_domain = []
        if 't' in cls.dims:
            used_domain += [np.array(input_time_domain)]
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
            cls.x_pts = swt.domain[cls.dims.index('x')][level]
            cls.shock_loc += [cls.x_pts[cls.shock_loc_indx]]
        if 'y' in cls.dims:
            cls.y_pts = swt.domain[cls.dims.index('y')][level]
            cls.shock_loc += [cls.y_pts[cls.shock_loc_indx]]
        if 'z' in cls.dims:
            cls.z_pts = swt.domain[cls.dims.index('z')][level]
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


#==================================================================================================
#
#   Fluid Structure Objects
#
#==================================================================================================

class turbulentShearMixingLayer:
    """
        This object contains all the data and methods required to generate and assess the behavior
    within it. References include:

    [43] Pantano, C. and Sarkar, S. (2002). *A study of compressibility effects in the high-speed 
            turbulent shear layer using direct simulation.* Journal of Fluid Mechanics. Vol 451, 
            pgs 329-371. DOI: 10.1017/S0022112001006978

    """

    def __init__(self, convective_MachNo=0.0, density_ratio=1.0):
        """
            Initialize the turbulent shear mixing layer. 

        Args:
            convective_MachNo (float, optional):    The convective Mach Number of the mixing layer. 
                                                    Defined as:
                                                    
                                                    M_c = \frac{\Delta u}{c_1 + c_2} [Eq 2.7, 43]

                                                    Defaults to 0.0, or an incompressible shear 
                                                    layer.

            density_ratio (float, optional):    The density ratio of the upper vs lower stream. 
                                                Defaults to 1.0. 

                                                s = \frac{\rho_2}{\rho_1} [Eq 2.9, 43]

        Attributes:

        M_c <= convectiveMachNo

        dens_ratio <= density_ratio

        Atwood_number: The Atwood number 

        """

        # Store mixing layer properties
        self.M_c = convective_MachNo
        self.dens_ratio = density_ratio

        # Calculate other values
        self.Atwood_number = ( self.dens_ratio - 1 ) / ( self.dens_ratio + 1 )

    def initialize_shearLayerProfile(cls, stream_velocity_difference, coordinates, coordinate_system=['x','y'], spanwise_coordinate=1 ):
        """
            This method allows the user to initialize the turbulent shear layer.

        Args:
            stream_velocity_difference [m/s] (float):   The difference in velocity between the 
                                                        freestreams. Defined as:

                                                        \Delta u=u_2 - u_1

            coordinates (float, NumPy 2D array):    The array that corresponds to the coordinates
                                                    of the shear layer. The first axis will pertain
                                                    to the axis of coordinates, then the second 
                                                    axis will pertain to the point sample.

            coordinate_sytem (string, list, optional):  The coordinate system that pertains to the
                                                        turbulent shear layer.

            spanwise_coordinate (int, optional):    The axis that pertains to the spanwise 
                                                    direction. Defaults to 1, which is entry #2 in
                                                    coordinate_system.

        """
        # Stream properties
        cls.DeltaU = stream_velocity_difference

        # Coordinates
        cls.coords = coordinates
        cls.coord_sys = coordinate_system
        cls.spanwise_coord = spanwise_coordinate
        if not len(cls.coord_sys)==cls.coords.shape[0]:
            if len(cls.coords.shape)==1:
                raise ValueError("The first axis of the coordinates is the number of points, make sure that coordinate entry is 2D NumPy array.")
            raise ValueError("Coordinate system contains a different number of axes than coordinates provided.")


    def shearLayerProfile_measure(cls, data, streamwise_velocity_key="U:x", density_key="rho", spanwise_velocity_key="U:y", 
                                  tke_key="k", turb_modeling=True, turb_viscosity_key="nut", turb_dissipation_key="omega", 
                                  temperature_key="T", C_delta=0.0384, ke_coeff = { "C_mu":0.09 }, 
                                  ko_coeff = { "beta*": 0.009, "sigma_k":0.85 }, viscosity_func="sutherland", 
                                  sutherland_coeffs={ "T_0":273.15, "S":110.4, "mu_0":17.16e-6 }, Temp=300 ):
        """
            This method allows the user to measure important data of the turbulent shear layer

        Args:
            data (dictionary):  The dictionary to pull the data from. Note that the first axis of
                                all the entries must pertain to time, then the second axis to the
                                sample points.

            streamwise_velocty_key (string, optional):  The key in the data that describes the
                                                        streamwise component of velocity. The 
                                                        default is "U:x".

            density_key (string, optional): The key in the data that describes the density. The
                                            the default is "rho".

            C_delta (float, optional):  The growth coefficient for the momentum thickness acquired
                                        by Dai et al. (2022). This produces an empirical value for
                                        momentum thickness growth rate for incompressible flow.

        """
        # Store data
        cls.data = data

        #=============================================================
        #
        #   Calculate Favre averaged data
        #
        #=============================================================

        cls.favre_data = {}
        cls.favre_data["u"], cls.favre_data["rho"] = favre_average( cls.data[density_key], cls.data[streamwise_velocity_key], return_rho_avg=True )
        cls.favre_data["v"] = favre_average( cls.data[density_key], cls.data[spanwise_velocity_key] )
        if not temperature_key==None:
            cls.favre_data["T"] = favre_average( cls.data[density_key], cls.data[temperature_key] )
        else:
            cls.favre_data["T"] = Temp
        if turb_modeling:
            cls.favre_data["k"] = favre_average( cls.data[density_key], cls.data[tke_key] )
            cls.favre_data["omega"] = favre_average( cls.data[density_key], cls.data[tke_key] )
            cls.favre_data["nut"] = favre_average( cls.data[density_key], cls.data[turb_viscosity_key] )

        #=============================================================
        #
        #   Calculate self-similar profile
        #
        #=============================================================

        # Calculate convective velocity
        cls.U_c = np.trapz( cls.favre_data["rho"] * cls.favre_data["u"], cls.coords[cls.spanwise_coord] ) / np.trapz( cls.favre_data["rho"] , cls.coords[cls.spanwise_coord] ) 

        # Calculate the momentum thickness
        u_tildes = cls.favre_data["u"]
        rho_avg = cls.favre_data["rho"]
        DU = np.abs( u_tildes[-1] - u_tildes[0] )
        #integral_val = np.trapz( rho_avg * ( cls.U_c - u_tildes ) * ( cls.U_c + u_tildes ), cls.coords[cls.spanwise_coord] )
        integral_val = np.trapz( rho_avg * ( cls.favre_data["u"][-1] - cls.favre_data["u"] ) * ( cls.favre_data["u"] - cls.favre_data["u"][0] ), cls.coords[cls.spanwise_coord] )
        cls.delta_theta = integral_val / ( np.mean( rho_avg ) * ( DU ** 2 ) )

        # Calculate the vorticity thickness
        cls.delta_omega = DU / np.max( np.gradient( u_tildes, cls.coords[cls.spanwise_coord] ) )

        # Calculate empirical incompressilbe growth rate
        cls.delta_theta_dot_incompressible = C_delta * ( 1 - u_tildes[0]/u_tildes[-1] ) * ( 1 + np.sqrt( rho_avg[0] / rho_avg[-1] ) ) / ( 2 * ( 1 + (u_tildes[0]/u_tildes[-1]) * np.sqrt( rho_avg[0] / rho_avg[-1] ) ) )
        
        #=============================================================
        #
        #   Calculate the TKE budget data
        #
        #=============================================================

        cls.tke_budget = {}

        # Calculate the production
        du_dy = np.gradient( cls.favre_data["u"], cls.coords[cls.spanwise_coord] )
        dv_dy = np.gradient( cls.favre_data["v"], cls.coords[cls.spanwise_coord] )
        if turb_modeling:
            cls.tke_budget["Production"] = 2 * cls.favre_data["nut"] * cls.favre_data["rho"] * ( 2 * ( du_dy ** 2 ) + ( dv_dy ** 2 ) )
            cls.tke_budget["Production"] -=  (2/3) * cls.favre_data["k"] * dv_dy

        # Calculate the dissipation
        if turb_modeling:
            #cls.tke_budget["Dissipation"] = -ko_coeff["beta*"] * cls.favre_data["rho"] * cls.favre_data["k"] * cls.favre_data["omega"]
            cls.tke_budget["Dissipation"] = -cls.favre_data["rho"] * ke_coeff["C_mu"] * ( cls.favre_data["k"]**2 ) / cls.favre_data["nut"]

        # Calculate the transport
        if viscosity_func.lower() in ["sutherland's law", "sutherlands law", "sutherland", "s"]:
            mu = sutherland_viscosity( cls.favre_data["T"], coefficients=sutherland_coeffs )
        if turb_modeling:
            cls.tke_budget["Transport"] = np.gradient( ( mu + ko_coeff["sigma_k"] * cls.favre_data["nut"] * cls.favre_data["rho"] ) * np.gradient( cls.favre_data["k"], cls.coords[cls.spanwise_coord] ), cls.coords[cls.spanwise_coord] )


    def initial_conditions(cls, stream_velocity_difference, mixingLayer_width, stream_temperatures, stream_Rs ):
        """
            Initialize the conditions for the mixing layer

        Args:
            stream_velocity_difference [m/s] (float):   The difference in velocity between the 
                                                        freestreams. Defined as:

                                                        \Delta u=u_2 - u_1
            
            mixingLayer_width [m] (float):  The width of the mixing layer at the beginning.

            stream_temperature [K] (float, NumPy 1D array): The freestream temperatures of the
                                                            different streams. 

            stream_Rs [J/kgK] (float, NumPy 1D array):  The freestream gas constants of the 
                                                        different streams.

        """
        # Initial dimensions
        cls.delta_theta0 = mixingLayer_width

        # Stream properties
        cls.DeltaU = stream_velocity_difference

        # Store the fluid conditions
        if not np.isscalar(stream_temperatures):
            cls.stream_T = np.array( [ stream_temperatures[0], stream_temperatures[-1] ] )
        else:
            cls.stream_T = np.array( [stream_temperatures, stream_temperatures] )

        if not np.isscalar( stream_Rs ):
            cls.stream_R = np.array( [stream_Rs[0], stream_Rs[-1]] )
        else:
            cls.stream_R = np.array( [stream_Rs, stream_Rs] )
        

    def domain_dimensions(cls, X_domain, N_x, Y_domain, N_y, Z_domain=None, N_z=0 ):
        """
            The dimensions of the domain that is getting set up.

        Args:
            X_domain (float, NumPy 1D Array):   The bounds for the x-domain.

            N_x (int):  The number of points in the domain.

            Y_domain (float, NumPy 1D Array):   The bounds for the y-domain.

            N_y (int):  The number of points in the domain.

            Z_domain (float, NumPy 1D Array):   The bounds for the z-domain. Defaults to None.

        Attributes:

        domain (float, list):   The domain. In the format of:

                                [dimension index][Point index]

        """
        # Initialize the domain
        cls.domain = []

        #
        #   Initialize the domains
        #
        X_domain = np.linspace( np.min( X_domain ), np.max( X_domain ), num=N_x )
        cls.domain += [X_domain]

        Y_domain = np.linspace( np.min( Y_domain ), np.max( Y_domain ), num=N_y )
        cls.domain += [Y_domain]

        if Z_domain:
            Z_domain = np.linspace( np.min( Z_domain ), np.max( Z_domain ), num=N_z )
            cls.domain += [Z_domain]


    def initialize_inletProfile(cls, average_density, streamwise_dimension=0, temperature_blending_factor="Reynolds", Lewis_no=1.0, U_offset = np.array( [ 0.0, 0.0, 0.0 ] ) ):
        """
            Initialize the inlet profile

        Args:
            average_density (float):    The average density of the streams.

            streamwise_dimension (int, optional):   The velocity component that defines the 
                                                    streamwise component. The following dimension
                                                    index will be the spanwise coordinate.

            temperature_blending_factor (optional): The hyperbolic tangent value to blend the 
                                                    profiles by. The valid options are:

                                                    "*Reynolds": Use the Reynolds analogy for 
                                                                blending where momentum and thermal
                                                                blending are proportional.

                                                    Not case sensitive.

        Attributes:

        u (float, NumPy 2D array): The velocity profile. Takes the format of:

                                    [(x, y, z) dimensions, N_points]

                                Where x is the streamwise direction by default. If there is no
                                z-dimension in cls.domain, then the z dimension will not be 
                                present.

        """
        N_points = cls.domain[streamwise_dimension+1].shape[0]

        # Store the streamwise dimension
        cls.streamwise = streamwise_dimension

        # Initialize the velocity profile
        cls.u = np.zeros( ( len(cls.domain), N_points ) )
        cls.u[streamwise_dimension,:] += -( cls.DeltaU / 2 ) * np.tanh( - cls.domain[ streamwise_dimension + 1 ] / ( 2 * cls.delta_theta0 ) )
        for i in range( len( cls.domain ) ):
            cls.u[i,:] += U_offset[i]

        # Store the average density
        cls.rho_avg = average_density

        # Initialize the density profile
        cls.rho = np.zeros( N_points )
        cls.rho = average_density * ( 1 - cls.Atwood_number * np.tanh( - cls.domain[ streamwise_dimension + 1 ] / ( 2 * cls.delta_theta0 ) ) )

        # Initialize the temperature
        cls.T = np.zeros( N_points )
        if not isinstance( temperature_blending_factor, str ):
            cls.T =  ( ( cls.stream_T[1] - cls.stream_T[0] ) / 2 ) * np.tanh( - cls.domain[ streamwise_dimension + 1 ] / temperature_blending_factor ) + np.mean( cls.stream_T )
        elif temperature_blending_factor.lower() in ["reynolds", "reynolds analogy", "re"]:
            cls.T =  ( ( cls.stream_T[1] - cls.stream_T[0] ) / 2 ) * np.tanh( - cls.domain[ streamwise_dimension + 1 ] / ( 2 * cls.delta_theta0 ) ) + np.mean( cls.stream_T )

        # Initialize the gas constants
        cls.R = np.zeros( N_points ) + np.mean( cls.stream_R )
        #cls.R = ( cls.stream_R[1] - cls.stream_R[0] ) * cls.T * Lewis_no / ( cls.stream_T[1] - cls.stream_T[0] )
        cls.R += np.nan_to_num( ( cls.stream_R[1] - cls.stream_R[0] ) * cls.T * Lewis_no / ( cls.stream_T[1] - cls.stream_T[0] ), nan=0 )

        # Initialize the pressure constants
        cls.p = np.zeros( N_points )
        cls.p = cls.rho * cls.R * cls.T

    def turbulence_inletProfile(cls, average_viscosity, R_xx_NormalizedPeak=[ 0.175, 0.134, 0.145 ], R_xx_Baseline=[ 0.025, 0.025, 0.010 ], epsilon_NormalizedPeak=1.2e-3, distribution_method="pdf", conversion_turbulenceModels=None, ke_coefficients={ "C_mu": 0.09 }, ko_coefficients={ "beta_star":0.09 }, SA_coefficients= { "C_v1": 7.1 }, minimum_dissipation=200.0e6, minimum_specDissipation=10e3, N_smoothing=1 ):
        """
            This method generates profiles of the turbulence according to the methods outlined in 
        [43]. 

        Args:
            average_viscosity [Pa*s] (float):   The density averaged viscosity of the two streams.
                                                Only used for the Spalart-Allmaras turbulence model
                                                conversion.

            R_xx_NormalizedPeak (list, optional): The peaks of the square root of two-point 
                                                    correlations normalized by \Delta u. Defaults 
                                                    to [ 0.175, 0.134, 0.145 ]. Produces turbulent
                                                    kinetic energy profile. Values from [43].

            R_xx_Baseline (list, optional): The baseline for the square root of two-point 
                                            correlations normalized by \Delta u. Default to 
                                            [0.025, 0.025, 0.010]. Produces the base of the TKE
                                            profile. Values from [43].

            distribution_method (str, optional):  The method to produce the distribution of 
                                                    turbulence statistics. The valid options are:

                                                "*pdf": Uses a probability density function to 
                                                        produce a distribution of the TKE, 
                                                        dissipation, etc. where the sqrt(variance),
                                                        or std dev, is 4\delta_\theta or 
                                                        \delta_\omega. There will be some offset
                                                        beyond this point.
                                                
            conversion_turbulenceModels (str, list):    A list of turbulence models to convert to.
                                                        If not necessary, leave as None. Valid
                                                        options are:

                                                        "k-omega": k-omega model

                                                        "SA": Spalart-Allmaras Model

        """
        

        # Initialize the base profiles
        cls.TKE = np.zeros_like( cls.rho )
        cls.dissipation = np.zeros_like( cls.TKE )

        #=============================================================
        #
        #   Set the baseline TKE & dissipation profile
        #
        #=============================================================
        if distribution_method.lower() in ["pdf", "probability", "bell", "p", "probability density function"]:
        
            # Pull the SciPy stats
            import scipy.stats as spst

            # Set the baseline values
            cls.TKE += ( np.linalg.norm( np.array( R_xx_Baseline ) ) * cls.DeltaU ) ** 2

            # Set the PDF base profile
            PDF_raw = spst.norm.pdf( cls.domain[cls.streamwise+1], loc=0, scale=cls.delta_theta0 )
            cls.PDF = PDF_raw / np.max( PDF_raw )

            # Rescale R_xx
            R_xx_rescale = np.array( R_xx_NormalizedPeak ) - np.array( R_xx_Baseline )
            
            # Add peak to TKE
            cls.TKE += cls.PDF * ( np.linalg.norm( R_xx_rescale ) * cls.DeltaU ) ** 2

            # Add distribution to epsilon
            disp_pdf = np.convolve( cls.PDF, np.ones( N_smoothing ) / N_smoothing, mode="same" )
            cls.dissipation += np.sqrt( disp_pdf / np.max(disp_pdf) ) * ( epsilon_NormalizedPeak * ( cls.DeltaU ** 3 ) / cls.delta_theta0 )

            cls.dissipation = np.maximum( cls.dissipation , minimum_dissipation * np.ones_like( len(cls.dissipation ) ) )
            #cls.dissipation = np.maximum( cls.dissipation , (ke_coefficients["C_mu"]**(3/4))*(cls.TKE**(3/2)) / cls.delta_theta0 )

        #=============================================================
        #
        #   Convert to different turbulence model
        #
        #=============================================================
        if not conversion_turbulenceModels is None:
            mut_max = cls.rho_avg * ke_coefficients["C_mu"] * ( np.max( cls.TKE ) ** 2 ) / np.max( cls.dissipation )
            cls.mut = cls.rho * ke_coefficients["C_mu"] * ( cls.TKE ** 2 ) / ( cls.dissipation )

            for model in conversion_turbulenceModels:

                if model.lower() in ["ko", "k-o", "komega", "k-omega", "menter", "wilcox"]:
                    cls.spec_dissipation = np.zeros_like( cls.dissipation )

                    spec_dissipation_max = np.max( cls.TKE ) / mut_max

                    #cls.spec_dissipation = cls.rho * cls.PDF * spec_dissipation_max + minimum_specDissipation
                    cls.spec_dissipation = ( cls.dissipation ) / ( cls.TKE * ko_coefficients["beta_star"] )
                    #cls.spec_dissipation = ( cls.mut / cls.rho ) / ( ko_coefficients["beta_star"] * cls.TKE )

                if model.lower() in ["spalart allmaras", "spalart-allmaras", "sa"]:

                    base_dynVisc = average_viscosity / cls.rho_avg

                    nu_tilde_0 = base_dynVisc * np.ones_like( cls.mut )

                    def nut_result( nu_tilde ):
                        """
                            This function produces the turbulent viscosity from the Spalart-
                        Allmaras model.

                        Remember that

                            \chi = \frac{\tilde{\nu}}{\nu}

                            and

                            f_{v1} = \frac{ \chi^3 }{ \chi^3 + C_{v1}^3 }

                        Args:
                            nu_tilde (float, NumPy 1D array): The array of nu-tilde values.

                        Returns
                            mu_t (float, NumPy 1D array):   The array of mu_t values.

                        """

                        chi = nu_tilde / base_dynVisc

                        f_v1 = ( chi**3 ) / ( ( chi**3 ) + SA_coefficients["C_v1"]**3 )

                        return cls.rho * nu_tilde * f_v1
                    
                    #print(f"Initial mu_t guesses:\t{nut_result(nu_tilde_0)}")
                    
                    def residuals( nu_tilde ):
                        return nut_result( nu_tilde ) - cls.mut
                    
                    #print(f"Initial residuals:\t{residuals(nu_tilde_0)}")
                    
                    import scipy.optimize as spopt

                    solved = spopt.least_squares( residuals, nu_tilde_0, gtol=None )
                    #cls.LSQ_solution = solved

                    cls.nu_tilde = solved.x

                        

class boundaryLayer:
    """
        This object holds the data and methods that describes a boundary layer.

    """

    def __init__(self, U_infty, nu, coordinates):
        """
            Initialize the boundary layer with needed data.

        Args:
            U_infty (float):    The freestream velocity.

            nu (float):     [m^2/s] The freestream viscosity.

            coordinates (float, NumPy NDArray): The wall-normal coordinate system for the boundary
                                                layer.

        """

        # Store the data
        self.U_infty = U_infty
        self.nu_infty = nu

        # Store the coordinates
        self.coordinates = coordinates


    def data_import(cls, data_dictionary, velocity_keys=["U:x", "U:y", "U:z"], pressure_key="p", temperature_key="T", density_key="rho", nu_T_key="nut" ):
        """
            Import the needed data into the boundaryLayer object.

        Args:
            data_dictionary (dictionary):   The input data dictionary.

            *Data Keys*: The keys of data dictionary that correspond to the following data. If not
                            to be stored, leave each as blank. The following are taken, and the 
                            default is:

            velocity_keys - ["U:x", "U:y", "U:z"] - keys for each component of velocity

            pressure_key - "p" - key for pressure

            temperature_key - "T" - key for temperature

            density_key - "rho" - key for density

            nu_T_key - "nut" - key for turbulent viscosity

        """
        cls.data = {}

        # Store all matching velocity keys individually
        cls.vel_keys = velocity_keys
        for k in velocity_keys:
            if k in data_dictionary:
                cls.data[k] = data_dictionary[k]


        # Import pressure
        if pressure_key:
            cls.pressure_key = pressure_key
            cls.data[pressure_key] = data_dictionary[pressure_key]

        # Import temperature
        if temperature_key:
            cls.temperature_key = temperature_key
            cls.data[temperature_key] = data_dictionary[temperature_key]

        # Import density
        if density_key:
            cls.density_key = density_key
            cls.data[density_key] = data_dictionary[density_key]

        # Import turbulent viscosity key
        if nu_T_key:
            cls.nu_T_key = nu_T_key
            cls.data[nu_T_key] = data_dictionary[nu_T_key]


    def boundaryLayerThickness_calculation(cls, threshold=0.99):
        """
            Calculate the boundary layer thicknesses.

        Args:
            threshold (float, optional):    The threshold that define boundary layer thickness. 
                                            Defaults to 0.99.

        """
        N_timeSteps = cls.data[cls.vel_keys[0]].shape[0]
        cls.N_timeSteps = N_timeSteps

        # Store the threshold
        cls.BL_threshold = threshold

        # Boundary layer thickness
        cls.delta = np.zeros( N_timeSteps )
        for i in range( N_timeSteps ):
            vel_ratios = (cls.data[cls.vel_keys[0]][i]/cls.U_infty -threshold)
            #print(f"vel ratios shape:\t{vel_ratios.shape}")
            cls.delta[i] = np.interp( 0, vel_ratios, cls.coordinates[1] )

        # Displacement thickness
        if hasattr( cls, "density_key" ):
            rho_us = cls.data[cls.density_key] * cls.data[cls.vel_keys[0]]
            rho_ue = cls.data[cls.density_key][-1] * cls.U_infty
        else:
            rho_us = cls.data[cls.vel_keys[0]]
            rho_ue = cls.U_infty
        cls.delta_star = np.trapz( (1-rho_us/rho_ue), cls.coordinates[1] )

        # Momentum thickness
        cls.theta = np.trapz( (rho_us/rho_ue)*(1-rho_us/rho_ue), cls.coordinates[1] )

        # Shape Factor
        cls.H = cls.delta_star / cls.theta

    def boundaryLayerProfile(cls, nu_wall=None, kappa=0.41, C_plus=5.0, isothermal=True, sutherland_coeffs={ "T_0":273.15, "S":110.4, "C_1":1.458e-6, "mu_0":17.16e-6, "R":287 } ):
        """
            This method calculate the boundary layer profile for the 

        Args:
            nu_wall (?, optional):  The method or value to calculate the viscosity at the wall. The
                                    valid options are:

                                    *None:  Use the freestream viscosity

                                    "sutherland" or "s":    Uses Sutherland's law

                                    <float> or <int>:   A known value for viscosity

            kappa (float, optional):    von Karman constant. Defaults to 0.41.

            C_plus (float, optional):   C^+ value for law of the wall. Defaults to 5.0, empirical
                                        value for smooth walls.

            isothermal (boolean, optional): Whether the boundary layer can be assumed isothermal. 
                                            Drives the interpolation method will be used for the
                                            viscosity profile in the boundary layer.

            sutherland_coeffs (float, dictionary, optional):    The Sutherland's Law coefficients.
                                                                Default values are for air. See:

                        https://www.cfd-online.com/Wiki/Sutherland%27s_law

        """

        #=============================================================
        #
        #   Calculate the wall viscosity
        #
        #=============================================================
        if nu_wall is None:
            nu_wall = cls.nu_infty

        elif isinstance( nu_wall, (int, float ) ):
            nu_wall = nu_wall

        elif nu_wall.lower() in ["sutherland", "s"]:
            method = nu_wall.lower()
            if isothermal:
                print( "WARNING: Sutherland method selected, but isothermal boundary layer selected. Isothermal overriden." )
            T_wall = cls.data[cls.temperature_key][:,0]
            mu_wall = sutherland_coeffs["mu_0"] * ( ( T_wall / sutherland_coeffs["T_0"] ) ** 1.5 ) * ( ( sutherland_coeffs["T_0"] + sutherland_coeffs["S"] ) / ( T_wall + sutherland_coeffs["S"] ) )
            if hasattr( cls, "density_key" ):
                nu_wall = mu_wall / cls.data[cls.density_key][:,0]
            else:
                rho_wall = cls.data[cls.pressure_key][:,0] / ( sutherland_coeffs["R"] * T_wall )
                nu_wall = mu_wall / rho_wall

        else:
            raise ValueError( "Wall viscosity method is invalid" )
        
        #=============================================================
        #
        #   Calculate BL parameters
        #
        #=============================================================

        # Calculate friction velocity
        cls.u_tau = np.zeros( cls.N_timeSteps )
        for i in range( cls.N_timeSteps ):
            tau_w = nu_wall * np.gradient( cls.data[cls.vel_keys[0]][i], cls.coordinates[1] )[0]
            cls.u_tau[i] = np.sqrt( tau_w )

        #=============================================================
        #
        #   Calculate profile
        #
        #=============================================================

        # Calculate law of the wall velocity
        cls.u_plus = cls.data[cls.vel_keys[0]] / cls.u_tau

        # Calculate the viscosity profiles
        if nu_wall is None or isinstance( nu_wall, (int, float ) ):
            if isothermal:
                DT = cls.coordinates[1][0] - cls.coordinates[1][-1]
                Dnu = nu_wall - cls.nu_infty
                nus = ( cls.coordinates[1] - cls.coordinates[1][-1] ) * ( Dnu / DT ) + cls.nu_infty
            else:
                DT = cls.data[cls.temperature_key][:,0] - cls.data[cls.temperature_key][:,-1]
                Dnu = nu_wall - cls.nu_infty
                nus = np.sqrt( ( cls.data[cls.temperature_key] - cls.data[cls.temperature_key][-1] )  * ( ( Dnu / DT ) ** 2 ) ) + cls.nu_infty
        elif method in ["sutherland", "s"]:
                Ts = cls.data[cls.temperature_key]
                mus = sutherland_coeffs["mu_0"] * ( ( Ts / sutherland_coeffs["T_0"] ) ** 1.5 ) * ( ( sutherland_coeffs["T_0"] + sutherland_coeffs["S"] ) / ( Ts + sutherland_coeffs["S"] ) )
                if hasattr( cls, "density_key" ):
                    nus = mus / cls.data[cls.density_key]
                else:
                    rhos = cls.data[cls.pressure_key] / ( sutherland_coeffs["R"] * Ts )
                    nus = mus / rhos
        cls.nus = nus


        # Calculate the law of the wall coordinates
        cls.y_plus = cls.coordinates[1] * cls.u_tau / nus








