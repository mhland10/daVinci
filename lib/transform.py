# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:38:46 2024

@author: m_hland10

TRANSFORM_LIB

This library contains the objects and functions required to transform the data
    between various domains. This will include the Fourier domain to view
    periodic or spectral behavior.
    
Version     Date        Description

0.0         2024/07/31  The initial version

"""

###############################################################################
#
# Import Required Libraries/Modules
#
###############################################################################

import numpy as np
import cupy as cp
import pywt
from numba import jit, prange
import copy

###############################################################################
#
# Spectral Functions
#
###############################################################################

def fourierTransformND( data , dt = 1.0 , N = None , fft_method = 'normal' , 
                       precision = np.float64 , target_processor = 'cpu' ):
    """
    This function takes the incoming NumPy matrix and takes the Fourier 
        transform. 

    Parameters
    ----------
    data :  The NumPy data matrix. The first axis (axis 0) will be the axis
                that the Fourier transform will be taken over.
                
    **dt :  The step size of the data to determine the frequencies that
                correspond to the Fourier data.
                
            The default is 1.0.
            
    **N :   The length of the Fourier transform domain axis.
    
            The default is None to allow the FFT function to define the length.
            
    **fft_method : [str]    The method of the FFT. The valid options are:
        
                    - *'normal' : Compute the FFT using a normal FFT.
                    
                    - 'real' : Compute a real FFT. The main difference is that
                                    the FFT will not consider leading values.
                                    
                    Not case sensitive.
                    
    **precision : [NumPy dtype] The precision to use in the calculations. The
                    valid options are:
                        
                    - *np.float64 : 64-bit float. All floats will be 64-bit
                                        precision. All complex values will be
                                        128-bit precision.
                                        
                    - np.float32 : 32-bit float. All floats will be 32-bit
                                        precision. All complex values will be
                                        64-bit precision.
                                        
    **target_processor : [str] The processor to target in the calculation. The
                            valid options are:
                                
                            - *'cpu' : The traditional CPU will be targeted.
                                        This will use Intel MKL architecture 
                                        via NumPy.
                                        
                            - 'gpu' : The GPU will be targeted. This will use
                                        CUDA (or potentially HIP) via CuPy.
                                        
                            Not case sensitive.

    Returns
    -------
    frequency_data :    The data of the frequencies that correspond to the FFT
                            data.
                            
    amplitude_data :    The resulting Fourier space data from the FFT.

    """
    
    if precision==np.float32:
        complex_precision=np.complex64
    elif precision==np.float64:
        complex_precision=np.complex128
    else:
        raise ValueError( "Invalid precision selected" )
        
    if target_processor.lower()=='cpu':
        xp=np
    elif target_processor.lower()=='gpu':
        xp=cp
    else:
        raise ValueError( "Invalid processor selected" )
        
    data_shape = np.shape( data )
    data_flat = np.moveaxis( np.reshape( data , ( data_shape[0] ,) + ( np.prod( data_shape[1:] ) ,) ) , 0 , -1 )
    
    if fft_method.lower()=='normal':
        if not N:
            amplitude_data_flat = xp.fft.fft( data_flat ).astype( complex_precision )
        else:
            amplitude_data_flat = xp.fft.fft( data_flat , n = N ).astype( complex_precision )
        fft_data_length = np.shape( amplitude_data_flat )[-1]
        frequency_data = xp.fft.fftfreq( data_shape[0] , d = dt ).astype( precision )
    elif fft_method.lower()=='real':
        if not N:
            amplitude_data_flat = xp.fft.rfft( data_flat ).astype( complex_precision )
        else:
            amplitude_data_flat = xp.fft.rfft( data_flat , n = N ).astype( complex_precision )
        fft_data_length = np.shape( amplitude_data_flat )[-1]
        frequency_data = xp.fft.rfftfreq( data_shape[0] , d = dt ).astype( precision )
    
    amplitude_data = np.reshape( np.moveaxis( amplitude_data_flat , -1 , 0 ) , ( fft_data_length ,) + data_shape[1:] )
        
    return frequency_data , amplitude_data

def fourierTransform1D( data , dt = 1.0 , N = None , fft_method = 'normal' , 
                       precision = np.float64 , target_processor = 'cpu' ):
    """
    This function takes the incoming singular dimensional NumPy array takes the 
        Fourier transform. 

    Parameters
    ----------
    data :  The NumPy data array. The single axis will be the axis that the 
                Fourier transform will be taken over.
                
    **dt :  The step size of the data to determine the frequencies that
                correspond to the Fourier data.
                
            The default is 1.0.
            
    **N :   The length of the Fourier transform domain axis.
    
            The default is None to allow the FFT function to define the length.
            
    **fft_method : [str]    The method of the FFT. The valid options are:
        
                    - *'normal' : Compute the FFT using a normal FFT.
                    
                    - 'real' : Compute a real FFT. The main difference is that
                                    the FFT will not consider leading values.
                                    
                    Not case sensitive.
                    
    **precision : [NumPy dtype] The precision to use in the calculations. The
                    valid options are:
                        
                    - *np.float64 : 64-bit float. All floats will be 64-bit
                                        precision. All complex values will be
                                        128-bit precision.
                                        
                    - np.float32 : 32-bit float. All floats will be 32-bit
                                        precision. All complex values will be
                                        64-bit precision.
                                        
    **target_processor : [str] The processor to target in the calculation. The
                            valid options are:
                                
                            - *'cpu' : The traditional CPU will be targeted.
                                        This will use Intel MKL architecture 
                                        via NumPy.
                                        
                            - 'gpu' : The GPU will be targeted. This will use
                                        CUDA (or potentially HIP) via CuPy.
                                        
                            Not case sensitive.

    Returns
    -------
    frequency_data :    The data of the frequencies that correspond to the FFT
                            data.
                            
    amplitude_data :    The resulting Fourier space data from the FFT.

    """
    
    if precision==np.float32:
        complex_precision=np.complex64
    elif precision==np.float64:
        complex_precision=np.complex128
    else:
        raise ValueError( "Invalid precision selected" )
        
    if target_processor.lower()=='cpu':
        xp=np
    elif target_processor.lower()=='gpu':
        xp=cp
    else:
        raise ValueError( "Invalid processor selected" )
        
    data_shape = np.shape( data )
    data_flat = data
    
    if fft_method.lower()=='normal':
        print("Using normal FFT")
        if not N:
            amplitude_data_flat = xp.fft.fft( data_flat ).astype( complex_precision )
        else:
            amplitude_data_flat = xp.fft.fft( data_flat , n = N ).astype( complex_precision )
        fft_data_length = np.shape( amplitude_data_flat )[-1]
        frequency_data = xp.fft.fftfreq( data_shape[0] , d = dt ).astype( precision )
    elif fft_method.lower()=='real':
        print("Using real FFT")
        if not N:
            amplitude_data_flat = xp.fft.rfft( data_flat ).astype( complex_precision )
        else:
            amplitude_data_flat = xp.fft.rfft( data_flat , n = N ).astype( complex_precision )
        fft_data_length = np.shape( amplitude_data_flat )[-1]
        frequency_data = xp.fft.rfftfreq( data_shape[0] , d = dt ).astype( precision )
    
    amplitude_data = amplitude_data_flat
        
    return frequency_data , amplitude_data

#==================================================================================================
#
#   DWT Functions
#
#==================================================================================================

def samplesToCoeffsDWT( N_samples, N_levels, support, verbosity=0 ):
    """
        This determines which samples of the original signal are represented by the coefficients
    for each level of the DWT.

        Note that this only pertains to a 1D DWT of multiple levels.

    Args:
        N_samples (int):    The number of samples in the original signal.

        N_levels (int): The number of levels in the DWT.

        support (int): The number of samples in the wavelet used in the DWT.

    Returns:
        coeff_list (list, int): A list of integers where each integer represents the number of coefficients at each level of the DWT. In the format:

                                [ level ][ coefficient index, sample index ]

    """

    coeff_list = []
    for i in np.arange( N_levels ):
        # The number of coefficients at this level
        N_samples_perLevel = np.ceil( N_samples / ( 2 ** i ) ).astype(int)

        # The number of samples represented by each coefficient
        N_samples_per_coeff = int( support * ( 2 ** i ) )

        # The number of coefficients that can be represented at this level
        N_coeffs_per_level = ( N_samples + support - 1 ) // 2

        if verbosity > 0:
            print( f"Level {i}: {N_samples_perLevel} samples, each coefficient representing {N_samples_per_coeff} samples, totaling {N_coeffs_per_level} coefficients" )

        #
        #   Produce the list of indices from the original signal that are represented by each coefficient
        # 
        coeff_list_atLevel = np.zeros( ( N_coeffs_per_level, N_samples_per_coeff ), dtype=int )
        if verbosity > 1:
            print( f"\tcoeff_list_atLevel.shape: {coeff_list_atLevel.shape}" )
        for j in np.arange( coeff_list_atLevel.shape[0] ):
            coeff_list_atLevel[j, :] = np.arange( j * 2, j * 2 + N_samples_per_coeff )-1


        # Check if the last coefficient goes over for over samples and shift as needed
        #"""
        if np.max( coeff_list_atLevel ) >= N_samples:
            difference = np.max( coeff_list_atLevel ) - N_samples
            if verbosity > 0:
                print( f"\tLast coefficient goes over the number of samples, shifting by {difference//2}" )
            coeff_list_atLevel = coeff_list_atLevel - difference // 2
        #"""

        coeff_list += [ coeff_list_atLevel ]


    return coeff_list

def lineDomainDWT( domain, N_levels, support, verbosity=0 ):
    """
        This function converts a 1D domain into the equivalent domain represented by the DWT coefficients.
    

    Args:
        domain (float, array):  The original domain to be converted.

        N_levels (int): The number of levels in the DWT.

        support (int): The number of samples in the wavelet used in the DWT.

    Returns:
        DWT_domain (float, list): The domain represented by the DWT coefficients. Will be in format:
                                    [ level ][ coefficient index ]

                                    
    """

    # Pull the coefficients for the domain
    coeffs = samplesToCoeffsDWT( domain.shape[0], N_levels, support )

    # Initialize the DWT domain
    DWT_domain = []
    for l in np.arange( N_levels ):
        if verbosity > 0:
            print(f"Level {l}:")
        DWT_domain_atLevel = np.zeros( coeffs[l].shape[0] )
        for c in np.arange( coeffs[l].shape[0] ):
            if verbosity > 0:
                print(f"\tCoefficient {c}:\t{coeffs[l][c]}")

            # Correct for lower bound
            filtered_coeffs = coeffs[l][c][coeffs[l][c]>=0]

            # Correct for upper bound
            filtered_coeffs = filtered_coeffs[filtered_coeffs<domain.shape[0]]

            # Add the domain represented by this coefficient
            domain_at_coeff = domain[ filtered_coeffs ]
            if verbosity > 1:
                print(f"\t\tDomain at coefficient {c}:\t{domain_at_coeff}")

            # Calculate the centroid for the domain at the coefficient
            DWT_domain_atLevel[c] = np.mean( domain_at_coeff )

        DWT_domain += [ DWT_domain_atLevel ]

    return DWT_domain




###############################################################################
#
#   Spectral Data Objects
#
###############################################################################

class SpectralData():
    
    def __init__( self , data , dt = 1.0 , N = None , fft_method = 'normal' , 
                           precision = np.float64 , target_processor = 'cpu' ):
        """
        This object provides a structure for all the spectral data to reside in
            to store and calculate it more efficiently.
            
        NOTE: The shapes of data must all be the same for using other methods
                of this object.

        Parameters
        ----------
        data :  The dictionary of data. The data within the entries must be 
                    numpy matrices of the same shape, with the first axis (axis
                    0) being the axis to take the Fourier transform on.
                    
        **dt :  The step size of the data to determine the frequencies that
                    correspond to the Fourier data.
                    
                The default is 1.0.
                
        **N :   The length of the Fourier transform domain axis.
        
                The default is None to allow the FFT function to define the 
                    length.
                
        **fft_method : [str]    The method of the FFT. The valid options are:
            
                        - *'normal' : Compute the FFT using a normal FFT.
                        
                        - 'real' : Compute a real FFT. The main difference is 
                                        that the FFT will not consider leading 
                                        values.
                                        
                        Not case sensitive.
                        
        **precision : [NumPy dtype] The precision to use in the calculations. 
                        The valid options are:
                            
                        - *np.float64 : 64-bit float. All floats will be 64-bit
                                            precision. All complex values will 
                                            be 128-bit precision.
                                            
                        - np.float32 : 32-bit float. All floats will be 32-bit
                                            precision. All complex values will 
                                            be 64-bit precision.
                                            
        **target_processor : [str] The processor to target in the calculation. 
                                The valid options are:
                                    
                                - *'cpu' : The traditional CPU will be 
                                            targeted. This will use Intel MKL 
                                            architecture via NumPy.
                                            
                                - 'gpu' : The GPU will be targeted. This will 
                                            use CUDA (or potentially HIP) via 
                                            CuPy.
                                            
                                Not case sensitive.

        Returns
        -------
        None.

        """
        
        variables = list( data.keys() )
        
        #
        # Store input data
        #
        self.dt = dt
        self.N = N
        self.fft_method = fft_method
        self.precision = precision
        self.target_processor = target_processor
        
        #
        # Perform Fourier transform
        #
        self.frequency_data = {}
        self.fourier_data = {}
        for i , v in enumerate( variables ):
            if len( np.shape( data[v] ) )>1:
                self.frequency_data[v] , self.fourier_data[v] = fourierTransformND( data[v] , dt = dt , N = N , fft_method = fft_method , precision = precision , target_processor = target_processor )
            else:
                self.frequency_data[v] , self.fourier_data[v] = fourierTransform1D( data[v] , dt = dt , N = N , fft_method = fft_method , precision = precision , target_processor = target_processor )
        
        #
        # Calculate ESD
        #
        self.energy_spectra = {}
        self.spectra = {}
        for i , v in enumerate( variables ):
            self.energy_spectra[v] = np.abs( self.fourier_data[v] ) ** 2
            self.spectra[v] = np.abs( self.fourier_data[v] )
            
        self.variables = variables
        
    def fullEnergy( cls ):
        """
        This method calculates all the energy possibilities possible in the 
            cross-value matrix.

        Returns
        -------
        None.

        """
        
        variables = list( cls.fourier_data.keys() )
        
        cls.full_energy_spectra = np.zeros( ( len( variables ) ,) + ( len( variables ) ,) + np.shape( cls.fourier_data[variables[0]] ) )
        cls.full_spectra = np.zeros( ( len( variables ) ,) + ( len( variables ) ,) + np.shape( cls.fourier_data[variables[0]] ) )
        
        for i , v in enumerate( variables ):
            for j , w in enumerate( variables ):
                cls.full_energy_spectra[i,j,...] = np.abs( cls.fourier_data[v] * cls.fourier_data[w] )
                cls.full_spectra[i,j,...] = np.sqrt( np.abs( cls.fourier_data[v] * cls.fourier_data[w] ) )
                
    def correlations( cls ):
        """
        This method calculates the full matrix of correlations.
        
        Requires that the real FFT method is used.

        Returns
        -------
        None.

        """
        
        variables = list( cls.fourier_data.keys() )
        
        dt_shape = np.shape( cls.fourier_data[variables[0]] )
        ifft_len = 2*(dt_shape[0]-1)
        if len( dt_shape )>1:
            corr_shape = ( ifft_len ,) + dt_shape[1:]
        else:
            corr_shape = ( ifft_len ,)
        correlation_shape = ( len( variables ) ,) + ( len( variables ) ,) + corr_shape
        cls.correlation = np.zeros( correlation_shape )
        
        for i , v in enumerate( variables ):
            for j , w in enumerate( variables ):
                fft_axis_length = np.shape( cls.fourier_data[v] )[0]
                corr_raw = np.roll( np.fft.irfft( cls.fourier_data[v] * np.conj( cls.fourier_data[w] ) , axis=0 ) , ifft_len//2 , axis = -1 ).astype( cls.precision )
                cls.correlation[i,j,...] = corr_raw / np.max( corr_raw[(ifft_len//2-2):(ifft_len//2+2)] , axis=0 )

class WaveletData():
    """
        This object contains all the algorithm necessary to perform the functions that pertain to 
    the wavelet transform.

    """
    def __init__(self, data, N_dims=1, layer_header=None, wavelet_family='db1' ):
        """
            Initialize the WaveletData object

        Args:
            data (float):   The dictionary of the Numpy matrices of data.

            N_dims (int, optional): The number of dimensions that will be used in the wavelet 
                                        transform. Defaults to 1.

                                    Note: N_dims>2 not currently implemented

            layer_header (string, optional):   The header of the layer that will be used in the
                                                calculation that determines the number of layers
                                                the DWT can produce. Defaults to None, which uses
                                                the first key.

            wavelet_family (string, optional): The family of the wavelet that will be used in the
                                                calculation of the maximum number of levels.
                                                Defaults to 'db1'.

        """
        

        # Store the data
        self.data = data

        # Store the number of dimensions
        if not N_dims:
            self.N_dims=len( np.shape( data[layer_header]) )
        else:
            self.N_dims = N_dims

        # Find and store the number of levels
        if not layer_header:
            self.max_levels = pywt.dwtn_max_level(self.data[list(data.keys())[0]].shape, pywt.Wavelet(wavelet_family))
        else:
            self.max_levels = pywt.dwtn_max_level(self.data[layer_header].shape, pywt.Wavelet(wavelet_family))

    def importCoordinates(cls, coordinates, time_steps ):
        """
            Import the coordinates of the data 

        Args:
            coordinates (list/array):   The coordinates of the data in space. Must be in the format
                                            [x, (?)y, (?)z] where some coordinates may be optional.

            time_steps (list/array):    The time steps that correspond to the data.

        """

        cls.coordinates = coordinates
        cls.time_steps = time_steps

    def waveletTransform(cls, families, keys=None, level=None, mode="symmetric", stackup="equivalent", stackup_levels=None,
                         t_axis=0, interpolator="linear", dwt_axis=-1 ):
        """
            Perform the wavelet transform 

        Args:
            families (string):  The list of families of the wavelets that will be used in the
                                    transform. Not case sensitive. Use pywt.wavelist() to find
                                    available options.

            keys (string, optional):    The list of keys to get the data over. If None, the 
                                            default, is given, then it will revert to the keys in
                                            cls.data.

            mode (string, optional):    The padding method that the wavelet transform methods are 
                                            using. For reference, see:

                                        https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes

                                        Not case sensitive.

            stackup (string, optional): The way that the different families will be handled. The 
                                            valid options are:

                                        - *"equivalent", "parallel", "equiv", "p", or "e":
                                                Each family of wavelets will be treated as at the 
                                                    same level.

                                        - "heirarchical", "heirarchy", or "h":
                                                Each family of wavelets will be subtracted from the
                                                    data as each wavelet is transformed down the 
                                                    list given to the method.

                                        The default is "equivalent". Not case sensitive.

            stackup_levels (int, optional): The list/array of levels to keep from the DWT 
                                                coefficients.

            t_axis (int, optional): The axis that defines which axis the time exists on. Default
                                        value is 0.

            interpolator (string, optional):    The interpolator that will be used if the 
                                                    heirarchical stackup is used. The valid options
                                                    are:

                                                - *"linear", "lin", or "l": Linear interpolator
                                                    that uses Delaunay triangulation.

        """
        # Initialize the coefficients dictionary
        cls.coeffs = {}

        # Initialize the levels to decompose on
        if not level:
            level = cls.max_levels
        print(f"Transforming over {level} levels")

        # Find the keys to use if not given
        if not keys:
            keys = cls.data.keys()
        print(f"Transforming for {keys}")

        # Store the DWT axis as needed
        cls.dwt_axis = dwt_axis

        # Find the stackup lengths if not given
        if not stackup_levels:
            stackup_levels = [1] * len( families )

        # Perform DWT for an equivalent list of wavelets
        if stackup.lower() in ["equivalent", "parallel", "equiv", "p", "e"]:
            # Perform the wavelet transform by family of wavelets, then keys, then dimensions
            for f in families:
                coeffs_hold = {}
                print(f"Running for wavelet family {f}")
                for d in keys:
                    print(f"\tTranforming for key {d}")
                    if cls.N_dims==1:
                        if level==1:
                            coeffs_hold[d] = pywt.dwt(cls.data[d], f, mode=mode.lower())
                        else:
                            coeffs_hold[d] = pywt.wavedec(cls.data[d], f, level=level, mode=mode.lower(), axis=dwt_axis)
                    elif cls.N_dims==2:
                        if level==1:
                            coeffs_hold[d] = pywt.dwt2(cls.data[d], f, mode=mode.lower())
                        else:  
                            coeffs_hold[d] = pywt.wavedec2(cls.data[d], f, level=level, mode=mode.lower())
                    else:
                        raise ValueError("Too many dimensions requested")
                cls.coeffs[f] = coeffs_hold
        
        elif stackup.lower() in ["heirarchical", "heirarchy", "h"]:
            # Perform the wavelet transform by family of wavelets, then keys, then dimensions while
            #   removing preceding wavelet transforms 
            print("Under construction")

        else:
            raise ValueError("Invalid stackup method selected.")
        

        # Store certain values
        cls.level = level
        cls.stackup = stackup
        cls.mode = mode
        cls.families = families

    def domains(cls, coords, level=None, coord_format="list"):
        """
            This method finds the domain for the wavelet transform.

        Args:
            coords (float - numpy ndarray):  The coordinates of the data in the discrete wavelet
                                                transform. Must be in a list or array. See 
                                                coord_fromat for formatting information.

            level (int, optional):  The number of levels in the DWT. Defaults to None.

            coord_format (string, optional):    The format of the coordinates. The valid options
                                                are:

                                                - *"list" or "l": The coordinates are in a list. 
                                                    Each entry in a list must be an array or list
                                                    that corresponds to the coordinates and shape
                                                    of the data along the entry's axis.

                                                - "mesh", "meshgrid", or "m":   The coordinates are
                                                    in a meshgrid format. The coordinates must be in the 
                                                    shape of the data.

        """

        # Set the number of levels
        if not level:
            level = cls.level

        #
        # Get the shape of the DWT data
        #
        cls.wt_shape = pywt.wavedecn_shapes( cls.data[list(cls.data.keys())[0]].shape, cls.families[0], mode=cls.mode )
        print(f"The shape of the DWT data is {cls.wt_shape}")

        # Get the shape of thet wavelets

        #
        # If the data has multiple dimensions
        #
        if cls.N_dims>1:
            print(f"**Using {cls.N_dims}D data**")

            #
            # Check the format of the coordinates
            #
            cls.cf = coord_format.lower()
            print(f"Coordinate format:\t{cls.cf}, type:\t{type(cls.cf)}")
            if cls.cf in ["list", "l"]:
                for i in range(len(coords)):
                    print(f"\tThe coordinates are shape {len(coords[i])}, while the data is shape {np.shape(cls.data[list(cls.data.keys())[0]])[i]}")
                    len_diff = len(coords[i])-np.shape(cls.data[list(cls.data.keys())[0]])[i]
                    if not len_diff==0:
                        raise ValueError(f"Coordinate {i} does not match the shape of the data.")
            elif cls.cf in ["mesh", "meshgrid", "m"]: 
                if not len(coords)==len(cls.data[cls.data.keys()[0]])==2:
                    raise ValueError("The meshgrid coordinates are not the same shape as the data.")
            else:
                raise ValueError("Invalid coordinate format selected.")
                
            #
            # Calculate the step size for the DWT data
            #
            steps = []
            raw_gradients = []
            if cls.cf in ["list", "l"]:
                raw_gradients += [np.gradient(coords[i]) for i in range(len(coords))]
            elif cls.cf in ["mesh", "meshgrid", "m"]:
                raw_gradients += [np.gradient(coords, axis=i) for i in range(len(coords.shape))]
            steps = [np.mean(raw_gradients[i]) for i in range(len(raw_gradients))]

            #
            # Calculate the steps in the domain for the DWT data
            #
            cls.level_steps = []
            if not level:
                level = cls.max_levels
            for i in range(level+1):
                if i<level:
                    cls.level_steps += [np.array(steps)*(2**(i+1))]
                elif i>=level:
                    cls.level_steps += [np.array(steps)*(2**(i))]
            cls.level_steps = cls.level_steps[::-1]
            print(f"The level steps are {cls.level_steps}")
            

            #
            # Calculate the domain for the DWT data
            #
            """
            cls.domain = []
            print(f"There are {level} levels")
            for i in range(level+1):
                if cls.cf in ["list", "l"]:
                    addition = []
                    for j in range( len( coords ) ):
                        beginning = coords[j][0]
                        print(f"{i} shape:\t{cls.wt_shape[i]}")
                        if i>0:
                            mult = cls.wt_shape[i]["dd"][j]
                        else:
                            mult = cls.wt_shape[i][j]
                        print(f"Multiplier:\t{ mult }")
                        end = coords[j][0] + (mult-0.5)*cls.level_steps[i][j]
                        print(f"Domain for coordinate {j} is in [{beginning}, {end}]")
                        addition += [np.arange( beginning, end, cls.level_steps[i][j] )]
                    cls.domain += [addition]
                elif cls.cf in ["mesh", "meshgrid", "m"]:
                    cls.domain += [ np.arange( np.moveaxis( np.moveaxis( coords[i], i, 0 )[0,...], 0, i ), 
                                            np.moveaxis( np.moveaxis( coords[i], i, 0 )[-1,...], 0, i )+cls.level_steps[i], 
                                            cls.level_steps[i] ) ]
            #"""
                    
            #   Calculate the support
            wavelets = []
            supports = []
            for i, f in enumerate( cls.families ):
                wavelets += [pywt.Wavelet( f )]
                supports += [wavelets[i].dec_len]
            cls.supports = supports

            #
            # Calculate the domain for the DWT data
            #
            cls.domain = []
            print(f"There are {level} levels")
            for j in range( len( coords ) ):
                print(f"\tj={j}")
                raw_domains = lineDomainDWT( coords[j], level, supports[0] )
                addition = raw_domains[::-1]
                print(f"\t\tAddition:\t{addition}")
                cls.domain += [addition]



        #
        # If the data is 1D
        #  
        else:

            #
            # Check the format of the coordinates
            #
            cls.cf = coord_format.lower()
            print(f"Coordinate format:\t{cls.cf}, type:\t{type(cls.cf)}")
            if cls.cf in ["list", "l"]:
                print(f"\tThe coordinates are shape {len(coords)}, while the data is shape {np.shape(cls.data[list(cls.data.keys())[0]])[0]}")
                len_diff = len(coords)-np.shape(cls.data[list(cls.data.keys())[0]])[cls.dwt_axis]
                #if not len_diff==0:
                #    raise ValueError(f"Coordinate 0 does not match the shape of the data.")
            elif cls.cf in ["mesh", "meshgrid", "m"]: 
                if not len(coords)==len(cls.data[cls.data.keys()[0]])==2:
                    raise ValueError("The meshgrid coordinates are not the same shape as the data.")
            else:
                raise ValueError("Invalid coordinate format selected.")
            
            #   Calculate the support
            wavelets = []
            supports = []
            for i, f in enumerate( cls.families ):
                wavelets += [pywt.Wavelet( f )]
                supports += [wavelets[i].dec_len]
            cls.supports = supports

            # Calculate the number of levels
            if not level:
                level = cls.max_levels

            #
            # Calculate the domains for the DWT data
            #
            cls.domains_DWT = {}
            for i, f in enumerate( cls.families ):
                raw_domains = lineDomainDWT( coords, level, supports[i] )
                raw_domains += [ raw_domains[-1] ]  # Add the last domain to the list
                cls.domains_DWT[f] = raw_domains[::-1]


            
        

    def convergence(cls, ):
        """
            In this method, we will be taking the different levels of the wavelet transform and 
        finding how well they fit to the original data.

        """

        # TODO: We need to actually make this work
        print("This method hasn't been finished yet")

                

#==================================================================================================
#
#   PCA Objects
#
#==================================================================================================
                
class Decomposition():
    
    def __init__( self, data, decomposition_axis=-1, precision=np.float64, target_processor='cpu', full_matrx=True ):
        """
        Calculates the decompositions for the incoming data dictionary along
            the specified axis.

        Parameters
        ----------
        data :  The dictionary of the NumPy matrix of data.
        
        decomposition_axis :    The axis to take the decomposition over.
        
        **precision : [NumPy dtype] The NumPy data type that corresponds to the 
                            precision of the desired calculation.
                            
        **target_processor : [str] The processor to target in the calculation. 
                                The valid options are:
                                    
                                - *'cpu' : The traditional CPU will be 
                                            targeted. This will use Intel MKL 
                                            architecture via NumPy.
                                            
                                - 'gpu' : The GPU will be targeted. This will 
                                            use CUDA (or potentially HIP) via 
                                            CuPy.
                                            
                                Not case sensitive. 
                                
        **full_matrx :  Compute the full matrices of the SVD.
        
                        The default value is True.

        Returns
        -------
        None.

        """
        
        
        
        variables = list( data.keys() )
        self.variables = variables
        
        self.X = {}
        self.Y = {}
        self.A = {}
        for i , v in enumerate( variables ):
            print(f"Data is shape {data[v].shape}")
            #
            # For matrix data
            #
            if len( data[v].shape )>1:
                d = np.moveaxis( data[v] , decomposition_axis , -1 )
                d_shape = np.shape( d )
                d_ = np.reshape( d , ( np.prod( d_shape[:-1] ) ,) + ( d_shape[-1] ) )
                self.X[v] = np.moveaxis( np.moveaxis( d_ , -1 , 0 )[:-1,...] , 0 , -1 )
                self.Y[v] = np.moveaxis( np.moveaxis( d_ , -1 , 0 )[1:,...] , 0 , -1 )

                self.A[v] = np.matmul( self.Y[v], np.linalg.pinv( self.X[v] ) )

            #
            # For array data
            #
            else:
                D = {}
                d = len(data[v])//2
                D[v] = np.array( [np.roll( data[v], -i ) for i in range(len(data[v]))] )
                self.X[v] = D[v][:d,:d]
                self.Y[v] = D[v][1:d+1,:d]
                
            
                self.A[v] = np.roll( np.matmul( self.Y[v] , np.linalg.pinv( self.X[v] ) ), -1, axis=-1 )
            
    def POD( cls ):
        """
        Calculates the Proper Orthogonal Decomposition from the Decomposition
            object.

        Returns
        -------
        None.

        """

        cls.energies = {}
        cls.POD_modes = {}
        for v in cls.variables:
            cls.POD_modes[v], cls.energies[v], _ = np.linalg.svd( cls.X[v] )
        
    def DMD( cls ):
        """
        Calculates the Dynamic Mode Decomposition from the Decomposition
            object.

        Returns
        -------
        None.

        """

        cls.eigenvalues = {}
        cls.DMD_modes = {}
        cls.POD_modes = {}
        cls.energies = {}
        for v in cls.variables:
            cls.POD_modes[v], cls.energies[v], _ = np.linalg.svd( cls.X[v] )
            if np.iscomplexobj( cls.POD_modes[v] ):
                A_tilda = np.matmul( cls.POD_modes[v].H, cls.A[v], cls.POD_modes[v] )
            else:
                A_tilda = np.matmul( cls.POD_modes[v].T, cls.A[v], cls.POD_modes[v] )
            Lambda, V_tilda = np.linalg.eig( A_tilda )
            cls.DMD_modes[v] = np.matmul( cls.POD_modes[v], V_tilda )
            cls.eigenvalues[v] = Lambda


        
        

