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

#==================================================================================================
#
#   Gradient Object
#
#==================================================================================================

def gradientCoefficients( nOrderDerivative , negSidePoints , posSidePoints , nOrderAccuracy ):
    """
    This function calculates the coefficients for a gradient calculation for a given set of
        conditions and template.

    Args:
        nOrderDerivative <int>:     The order of the derivative that will be calculated.
        
        negSidePoints <int>:    The number of points from the reference points on the LHS or
                                    approaching negative infinity.

        posSidePoints <int>:    The number of points form the reference points on the RHS or 
                                    approaching positive infinity. 

        nOrderAccuracy <int>:   The 

    Returns:
        coeffs [float]: The array of the coefficients to the function values for the gradient that
                            is calculated.
    
    """

    n_template = negSidePoints + posSidePoints + 1
    # Calcualte the number of points in the template
    if nOrderAccuracy > n_template+1 :
        raise ValueError( "Too few points input for the order of derivative" )
    #print(  "There are {x} points in the template: {y} on the LHS and {z} on the RHS".format( x = n_template , y = negSidePoints , z = posSidePoints )   )

    points = [ x for x in range( -negSidePoints , posSidePoints + 1 ) ]
    nOrderAccuracy = len( points )
    #print( "With points:\t" + str( points ) )
    
    taylor_series_coeffs = np.ones( ( nOrderAccuracy ,) + ( nOrderAccuracy ,) )
    # Generate a matrix that contains the coefficients
    for i in range( nOrderAccuracy ):
        for j in range( nOrderAccuracy ):
            p = points[j]
            #print( "For i={x} and j={y}".format( x = i , y = j ) )
            #print( "\tp is "+str(p) )
            #c = ( p ** i ) / np.max( [ spsp.factorial( i ) , 1 ] )
            fracs = np.asarray( [ np.math.factorial( i ) , 1 ] ).max()
            #print( "\tfactorial is " + str( fracs ) )
            c = ( p ** i ) / fracs
            #print( "\tThe coefficient is {x}".format( x = c ) )
            taylor_series_coeffs[i,j] = c
        print(" ")
    # and fill out this matrix
    #print( "\nTaylor series coefficients are:\n"+str( taylor_series_coeffs ) )

    b = np.zeros( nOrderAccuracy )
    b[nOrderDerivative] = 1
    #print("b vector:\t"+str(b))
    coeffs = np.linalg.solve( taylor_series_coeffs , b )
    # Calculate the coefficients from the Taylor series coefficients
            
    return coeffs

class numericalGradient:

    def __init__( self , derivativeOrder , template ):
        """

        This object contains the data pertaining to a numerical gradient

        Args:
            derivativeOrder (int):  The order of the derivative that will be used.

            template ((int)):       The terms in the template that will be used for the
                                        gradient. This will be a tuple of (2x) entries.
                                        The first entry is the number of entries on the 
                                        LHS of the reference point. The second/last 
                                        entry is the number of entries on the RHS of
                                        the reference point.

        Attributes:

            derivativeOrder <-  Args of the same

            template        <-  Args of the same

            coeffs [float]: The coefficients of the numerical gradient according to the
                                template that was put in the object.

        """

        if len( template ) > 2:
            raise ValueError( "Too many values in \"template\". Must be 2 entries." )
        elif len( template ) < 2:
            raise ValueError( "Too few values in \"template\". Must be 2 entries." )

        self.derivativeOrder = derivativeOrder
        self.template = template

        self.coeffs = gradientCoefficients( self.derivativeOrder , self.template[0] , self.template[1] , self.derivativeOrder )
        self.coeffs_LHS = gradientCoefficients( self.derivativeOrder , 0 , self.template[0] + self.template[1] , self.derivativeOrder )
        self.coeffs_RHS = gradientCoefficients( self.derivativeOrder , self.template[0] + self.template[1] , 0 , self.derivativeOrder )

    def formMatrix( cls , nPoints , acceleration = None ):
        """

        Form the matrix that calculates the gradient defined by the object. Will follow
            the format:

        [A]<u>=<u^(f)>, where f is the order of the derivative, representing such.

        It will store the [A] is the diagonal sparse format provided by SciPy.sparse

        Args:
            nPoints (int):  The number of points in the full mesh.

            accelerateion (str , optional):    The acceleration method to improve the performance of calculating the
                                        matrix. The valid options are:

                                    - *None :    No acceleration

        Attributes:
            gradientMatrix <Scipy DIA Sparse>[float]:   The matrix to find the gradients.

        
        """

        import scipy.sparse as spsr

        #
        # Place the data into a CSR matrix
        #
        row = []
        col = []
        data = []
        for j in range( nPoints ):
            #print("j:\t{x}".format(x=j))
            row_array = np.zeros( nPoints )
            if j < cls.template[0]:
                row_array[j:(j+len(cls.coeffs_LHS))] = cls.coeffs_LHS
            elif j >= nPoints - cls.template[0]:
                row_array[(j-len(cls.coeffs_RHS)+1):(j+1)] = cls.coeffs_RHS
            else:
                row_array[(j-cls.template[0]):(j+cls.template[1]+1)] = cls.coeffs
            #print("\trow array:\t"+str(row_array))

            row_cols_array = np.nonzero( row_array )[0]
            row_rows_array = np.asarray( [j] * len( row_cols_array ) , dtype = np.int64 )
            row_data_array = row_array[row_cols_array]
            #print( "\tColumns of non-zero:\t"+str(row_cols_array))
            #print( "\tData of non-zero:\t"+str(row_data_array))

            row += list( row_rows_array )
            col += list( row_cols_array )
            data += list( row_data_array )

        cls_data = np.asarray( data )
        cls_row = np.asarray( row , dtype = np.int64 )
        cls_col = np.asarray( col , dtype = np.int64 )
        #print("\nFinal Data:\t"+str(cls_data))
        #print("Final Rows:\t"+str(cls_row))
        #print("Final Columns:\t"+str(cls_col))

        gradientMatrix_csr = spsr.csr_matrix( ( cls_data , ( cls_row , cls_col ) ) , shape = ( nPoints , nPoints ) )

        #
        # Transfer data to DIA matrix
        #
        cls.gradientMatrix = gradientMatrix_csr.todia()

    def gradientCalc( cls , x , f_x , method = "native" ):
        """

        This method calculates the gradient associated with the discrete values entered into the method.

        Args:
            x [float]:      The discrete values in the domain to calculate the derivative over.

            f_x [float]:    The discrete values in the range to calculate the derivative over.
            
            method (str, optional):     The method of how the gradient will be calculated. The valid options
                                            are:

                                        - *"native" :   A simple matrix multiplication will be used.

                                        - "loop" :  Loop through the rows. Will transfer the matrix to CSR
                                                        to index through rows.

                                        Not case sensitive.

        Returns:
            gradient [float]:   The gradient of the function that was input to the method.

        """

        if len( f_x ) != len( x ):
            raise ValueError( "Lengths of input discrete arrays are not the same." )

        gradient = np.zeros( np.shape( f_x ) )
        dx = np.mean( np.gradient( x ) )
        cls.formMatrix( len( f_x ) )

        if method.lower()=='loop':

            for i , x_i in enumerate( x ):
                #print("i:\t{x}".format(x=i))
                csr_gradient = cls.gradientMatrix.tocsr()
                row = csr_gradient.getrow(i)
                #print("\tRow:\t"+str(row))
                #print("\tf(x):\t"+str(f_x))
                top = row * f_x
                #print("\tTop Portion:\t"+str(top))
                #print("\tdx:\t"+str(dx))
                gradient[i] = top / dx

        elif method.lower()=='native':

            gradient = cls.gradientMatrix.dot( f_x ) / dx

        else:

            raise ValueError( "Invalid method selected" )
        
        return gradient            



    



