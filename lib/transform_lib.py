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
    def __init__(self, data, N_dims=1):
        """
            Initialize the WaveletData object

        Args:
            data (float):   The dictionary of the Numpy matrices of data.

            N_dims (int, optional): The number of dimensions that will be used in the wavelet 
                                        transform. Defaults to 1.

                                    Note: N_dims>2 not currently implemented

        """
        

        # Store the data
        self.data = data

        # Store the number of dimensions
        self.N_dims = N_dims

    def waveletTransform(cls, families, keys=None ):
        """
            Perform the wavelet transform 

        Args:
            families (string):  The list of families of the wavelets that will be used in the
                                    transform. Not case sensitive. Use pywt.wavelist() to find
                                    available options.

            keys (string, optional):    The list of keys to get the data over. If None, the 
                                            default, is given, then it will revert to the keys in
                                            cls.data.

        """
        cls.coeffs = {}

        if not keys:
            keys = cls.data.keys()

        for f in families:
            coeffs_hold = {}
            print(f"Running for wavelet family {f}")
            for d in keys:
                print(f"\tTranforming for key {d}")
                if cls.N_dims==1:
                    coeffs_hold[d] = pywt.dwt(cls.data[d], f)
                elif cls.N_dims==2:
                    coeffs_hold[d] = pywt.dwt2(cls.data[d], f)
                else:
                    raise ValueError("Too many dimensions requested")
            cls.coeffs[f] = coeffs_hold
                

#==================================================================================================
#
#   PCA Objects
#
#==================================================================================================
                
class Decomposition():
    
    def __init__( self , data , decomposition_axis , precision = np.float64 , target_processor = 'cpu' , full_matrx = True ):
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
        
        self.A = {}
        self.A_ = {}
        self.A_pinv = {}
        self.phi = {}
        self.sigs = {}
        self.v = {}
        for i , v in enumerate( variables ):
            d = np.moveaxis( data[v] , decomposition_axis , -1 )
            d_shape = np.shape( d )
            d_ = np.reshape( d , ( np.prod( d_shape[:-1] ) ,) + ( d_shape[-1] ) )
            self.A[v] = np.moveaxis( np.moveaxis( d_ , -1 , 0 )[:-1,...] , 0 , -1 )
            self.A_[v] = np.moveaxis( np.moveaxis( d_ , -1 , 0 )[1:,...] , 0 , -1 )
            
            self.phi[v] , self.sigs[v] , vh = np.linalg.svd( self.A[v] , full_matrices = full_matrx )
            self.v[v] = np.conj( v.T )
            
            self.A_pinv[v] = np.linalg.pinv( self.A[v] )
            
    def POD( cls ):
        """
        Calculates the Proper Orthogonal Decomposition from the Decomposition
            object.

        Returns
        -------
        None.

        """
        
        cls.POD_modes = cls.phi
        
    def DMD( cls ):
        """
        Calculates the Dynamic Mode Decomposition from the Decomposition
            object.

        Returns
        -------
        None.

        """
        
        

