# -*- coding: utf-8 -*-
"""

**PROJECT DA VINCI**

by Lightning Custom Shop

Created on Sun May 12 18:28:34 2024

@author: mtthl

This module contains all the functions and objects of Project da Vinci.

Project da Vinci is a LCS project to understand financial markets via the same
    math and patterns used to understand quantum mechanics and fluid 
    turbulence. This will be based on the proposed understanding by Mandelbrot.
    
Version     Date        Description

0.0         2024/05/12  The initial version

"""

###############################################################################
#
# Import Modules
#
###############################################################################

#
# Import da Vinci Libararies
#
import sys, os
lib_path = os.path.abspath(r"A:\daVinci\lib")
sys.path.append( lib_path )
from transform_lib import *


#
# Import Environment Packages
#
import numpy as np
import yfinance as yf

###############################################################################
#
# Financial Data Functions
#
###############################################################################

def ticker_datapull( tickerstr , st_date = '2000-01-01' , en_date = '2023-12-31' , interval_l = 'daily' ):
    """
    This function pulls the data for the input ticker symbol for the input
        dates.

    Parameters
    ----------
    tickerstr :     The string of the ticker symbol. Not case sensitive.
    
    **st_date :     The string of the date where the data will start. Must be
                        in the format YYYY-MM-DD.
    
                    The default is '2000-01-01'.
                    
    **en_date :     The string of the date where the data will end. Must be in
                        the format YYYY-MM-DD.
                        
                    The default is '2023-12-31'.
                    
    **interval_l :  The interval length in time of the data. Not case 
                        sensitive. The valid values are:
                            
                    'daily' :   Daily intervals
                    
                    'monthly' : Monthly intervals
                    
                    'weekly' :  Weekly intervals
                    
                    'XX-minute' :   Intervals by minute. The valid values of 
                                        XX are:
                                            
                                    - 01
                                    - 05
                                    - 30
                                    
                    'XX-hour' :     Intervals by hour. The valid values of XX
                                        are:
                                            
                                    - 01
                                    - 02
                                    
                    The default value is 'daily'. One should be very careful 
                        using minute or hourly data due to the size.

    Returns
    -------
    
    data :          The list of the dataframe of the input ticker symbol.

    """
    
    if interval_l.lower()=='daily':
        int_l = '1d'
    elif interval_l.lower()=='monthly':
        int_l = '1mo'
    elif interval_l.lower()=='weekly':
        int_l = '1wk'
    elif interval_l[-6:].lower()=='minute':
        if int(interval_l[:1])==1:
            int_l = '1m'
        elif int(interval_l[:1])==5:
            int_l = '5m'
        elif int(interval_l[:1])==30:
            int_l = '30m'
        else:
            raise Exception( "Invalid Minute Interval Input" )
    elif interval_l[-4:].lower()=='hour':
        if int(interval_l[:1])==1:
            int_l = '1h'
        elif int(interval_l[:1])==2:
            int_l = '2h'
        else:
            raise Exception( "Invalid Hour Interval Input" )
    else:
        raise Exception( "Invalid Interval Called" )
    
    data = yf.download( tickerstr , start = st_date , end = en_date , interval = int_l )
        
    return data

def simpleMovingAverage( price_data , movingaveragelength ):
    """
    This function provides the simple moving average data for the input price
        data and moving average length.

    Parameters
    ----------
    price_data :            The input price data.
    
    movingaveragelength :   The term of the moving average.
    
    Returns
    -------
    SMA_price_data :        The SMA price data.

    """
    
    price_data_flat = price_data
    
    if movingaveragelength >= len( price_data_flat ):
        raise Exception( "Moving Average Length is too long for the input data" )
        
    rectangle = np.ones( movingaveragelength ) / movingaveragelength
    
    if len( price_data.shape ) > 1 :
        SMA_price_data = np.zeros( price_data_flat.shape )
        for i in range( price_data_flat.shape[-1] ):
            SMA_price_data = np.convolve( price_data_flat[:,i] , rectangle )
    else:
        SMA_price_data = np.convolve( price_data_flat , rectangle )
    
    return SMA_price_data

def price_velocity( ticker_data , price_time = 'C' , time_interval = 'daily' , movingaveragelength = 50 , start_date = '2000-01-01' , end_date = '2023-12-31' ):
    """
    This function calculates the velocity, or time gradient, of the price for 
        the input ticker symbol. There are two forms of the velocity. The first
        is the raw velocity. The second is the velocity normalized to the
        moving average to compensate for long-term changes over time.

    Parameters
    ----------
    ticker_data :   The string of the ticker symbol.
    
    **price_time :  Where in the market time the price data will be considered
                        from. There are four input characters that define the
                        price:
                            
                    - 'O' :     Open price
                    - 'C' :     Close price
                    - 'L' :     Low price
                    - 'H' :     High price
                    - 'A' :     Adjusted close price
                    
                    From these characters, there are the following valid
                        options as to how the price is formulated:
                            
                    - 'X' :     Single point price
                    - 'XX' :    Two points that are averaged together to form
                                    the price
                    - 'O-C' :   Forms the velocity of the price by only the 
                                    difference from open to close.
                    - 'OCHL' :  The average of the O, C, H, and L prices.
                    
                    Not case sensitive. The default is 'C'.
                    
    **time_interval :   The interval length in time of the data. Not case 
                        sensitive. The valid values are:
                            
                    'daily' :   Daily intervals
                    
                    'monthly' : Monthly intervals
                    
                    'weekly' :  Weekly intervals
                    
                    'XX-minute' :   Intervals by minute. The valid values of 
                                        XX are:
                                            
                                    - 01
                                    - 05
                                    - 30
                                    
                    'XX-hour' :     Intervals by hour. The valid values of XX
                                        are:
                                            
                                    - 01
                                    - 02
                                    
                    The default value is 'daily'. One should be very careful 
                        using minute or hourly data due to the size.
                        
    ***movingaveragelength :    The length of the moving avearge to be
                                normalized to.
                                
                                The default is 50.

    Returns
    -------
    U :             The velocity of the price.
    
    U_normalized :  The velocity of the price as normalized to the moving 
                        average.
                        
    delta_d_length :    The length of the original data that was lost in the 
                            moving average calculation.

    """
    
    ticker_raw_data = ticker_datapull( ticker_data , st_date = start_date , en_date = end_date , interval_l = time_interval )
    
    if len( price_time ) == 1:
        if price_time.upper() == 'O':
            col_header = 'Open'
        elif price_time.upper() == 'C':
            col_header = 'Close'
        elif price_time.upper() == 'L':
            col_header = 'Low'
        elif price_time.upper() == 'H':
            col_header = 'High'
        elif price_time.upper() == 'A':
            col_header = 'Adj Close'
        else:
            raise Exception( "Invalid Column" )
        price_data = ticker_raw_data[ col_header ].values
    elif len( price_time ) == 2:
        price_data_raw = np.zeros( ( len( ticker_raw_data['C'].values ) , 2 ) )
        for i , c in enumerate( col_header.upper() ):
            if price_time.upper() == 'O':
                col_header = 'Open'
            elif price_time.upper() == 'C':
                col_header = 'Close'
            elif price_time.upper() == 'L':
                col_header = 'Low'
            elif price_time.upper() == 'H':
                col_header = 'High'
            elif price_time.upper() == 'A':
                col_header = 'Adj Close'
            else:
                raise Exception( "Invalid Column" )
            price_data_raw[:,i] = ticker_raw_data[ col_header ].values
        price_data = np.mean( price_data_raw , axis = -1 )
    elif len( price_time ) == 3:
        if price_time.upper() == 'O-C':
            price_data = ticker_raw_data['Close'].values - ticker_raw_data['Open'].values
        else:
            raise Exception( "Invalid Columns input for three values" )
    elif len( price_time ) == 4:
        if all( x in price_time for x in 'OCHL' ):
            price_data = ( ticker_raw_data['Close'].values + ticker_raw_data['Open'].values + ticker_raw_data['Low'].values + ticker_raw_data['High'].values ) / 4
        else:
            raise Exception( "Invalid Columns input for four values" )
    else:
        raise Exception( "Number of Columns not supported" )
        
    if time_interval.lower()=='daily':
        dt = 1
    elif time_interval.lower()=='monthly':
        dt = ( 365.25 / 12 )
    elif time_interval.lower()=='weekly':
        dt = ( 365.25 / 52 )
    elif time_interval[-6:].lower()=='minute':
        if int(time_interval[:1])==1:
            dt = 1 / ( 24 * 60 )
        elif int(time_interval[:1])==5:
            dt = 5 / ( 24 * 60 )
        elif int(time_interval[:1])==30:
            dt = 30 / ( 24 * 60 )
        else:
            raise Exception( "Invalid Minute Interval Input" )
    elif time_interval[-4:].lower()=='hour':
        if int(time_interval[:1])==1:
            dt = 1 / ( 24 )
        elif int(time_interval[:1])==2:
            dt = 2 / ( 24 )
        else:
            raise Exception( "Invalid Hour Interval Input" )
    else:
        raise Exception( "Invalid Interval Called" )
    
    
    U = np.gradient( price_data ) / dt
    U_SMA = simpleMovingAverage( U , movingaveragelength )
    
    
    #
    # Crop U and SMA
    #
    convolution_length = len( U ) + movingaveragelength - 1
    validity_length = len(U) - movingaveragelength + 1
    delta_c_length = ( convolution_length - validity_length )
    delta_d_length = len( U ) - validity_length
    U = U[delta_d_length//2:-delta_d_length//2]
    U_SMA = U_SMA[delta_c_length//2:-delta_c_length//2]
    
    U_normalized = U / U_SMA
    
    return U , U_normalized , delta_d_length

def correlation_1D( u_data , v_data ):
    """
    Calculates the correlation along the ONLY axis (axis 0) of "u_data" and 
        "v_data" and returns the correlation between the two.

    Parameters
    ----------
    u_data :    The 1D array of the u data.
    
    v_data :    The 1D array of the v data.

    Returns
    -------
    correlation_data :  The correlation between the two sets of data.

    """
    
    u_hat = np.fft.rfft( u_data )
    v_hat = np.fft.rfft( v_data )
    
    correlation_data = np.fft.irfft( np.fft.rfft( u_data ) * np.conj( np.fft.rfft( v_data ) ) )
    
    correlation_data = np.roll( correlation_data , len( correlation_data ) // 2 )
    
    correlation_data = correlation_data / np.max( correlation_data )
    
    return correlation_data

def correlation_ND( u_data , v_data ):
    """
    Calculates the correlation along the first axis (axis 0) of "u_data" and 
        "v_data" and returns the correlation between the two.

    Parameters
    ----------
    u_data :    The ND matrix of the u data. Note that the axis to take the 
                    correlation over must be axis 0.
                    
    v_data :    The ND matrix of the v data. Note that the axis to take the 
                    correlation over must be axis 0.

    Returns
    -------
    correlation_data :  The correlation between the two sets of data.

    """
    
    u_shape = u_data.shape
    new_u_shape = ( u_shape[0] , np.prod( u_shape[1:] ) )
    v_shape = v_data.shape
    new_v_shape = ( v_shape[0] , np.prod( v_shape[1:] ) )
    # Find the new shapes of the input data

    u_data_2d = u_data.reshape(new_u_shape)
    v_data_2d = v_data.reshape(new_v_shape)
    # Put the data in the new shape
    
    u_hat_2d = np.fft.rfft( u_data_2d.T )
    v_hat_2d = np.fft.rfft( v_data_2d.T )
    # Put the data into the Fourier space
    
    correlation_data = np.fft.irfft( u_hat_2d * np.conj( v_hat_2d ) )
    # Calculate the correlation
    
    correlation_shape = correlation_data.shape

    correlation_data = np.roll( correlation_data , correlation_shape[-1] // 2 , axis = -1 ).T    
    
    correlation_data = correlation_data / np.max( correlation_data , axis = 0 )
    
    return correlation_data

###############################################################################
#
# Objects
#
###############################################################################

