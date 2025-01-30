"""

sciDataRead

@author:    Matthew Holland

@email:     matthew.holland@my.utsa.edu

Date:       2025/01/29

    This file contains the objects and functions necessary for scientific data reading and storage.
This includes:

    - vtk
    - h5

For the following standard reading softwares:

    - paraview
    - OpenFOAM
    - ConvergeCFD

Version Date        Description

0.0     2025/01/29  Original Version

"""

#==================================================================================================
#
#   Imports
#
#==================================================================================================

import os, glob
import numpy as np
import paraview.simple as pasi
import vtk.util.numpy_support as nps
import vtk
import sys
import pandas as pd
from natsort import natsorted

#===================================================================================================
#
#   Paraview Post-Processing Objects
#
#===================================================================================================

class rake:
    """
    This object is a rake of points that allows the user to draw data from the datafiles to draw
        the wanted data.
    
    """

    def __init__( self , points , datafile , file_format="vtk" ):
        """
        Initialize the rake object according to the inputs to the file.

        The data will be stored in a Paraview-native format.

        Args:

            points ((arrays/lists)):    The tuple of arrays or lists that contain the points of
                                            the rakes. The order will be (x, y, z). All three
                                            dimensions are required.

            datafile (string):  The datafile with the CFD data.

            file_format (string, optional): The file format that will be used. The valid options 
                                                are:

                                            - *"vtk" - The default *.vtk output as OpenFOAM 
                                                        produces
                                            
                                            - "h5" - The *.h5 output that is defined by the 

                                            - None - Take the file format from the "datafile"
                                                        argument.

        Attributes:

            ext_points [list]:  The externally defined points from "points" re-formatted into a
                                    Paraview-friendly format.

            
        
        """

        # Check the number of dimensions
        if not len( points ) == 3:
            raise ValueError( "Not enough dimensions in points. Make sure three (3) dimensions are present.")
        
        # If the file format is inferred, find what it is
        if file_format is None:
            ext = os.path.splitext(datafile)[-1].lower()
            if ext == ".vtk":
                file_format = "vtk"
            elif ext == ".h5":
                file_format = "h5"
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
        
        # Load the data from the *.vtk files
        # Expand wildcard pattern into actual file list with natural sorting
        file_list = natsorted(glob.glob(datafile))  # Use natsorted instead of sorted
        if not file_list:
            raise FileNotFoundError(f"No files found matching {datafile}")
        if file_format.lower()=="vtk" or file_format.lower()==".vtk":
            data = pasi.OpenDataFile( file_list )
        elif file_format.lower()=="h5" or file_format.lower()==".h5":
            data = pasi.CONVERGECFDReader(FileNames=file_list)
        else:
            raise ValueError(f"Unsupported file format:\t{file_format}")


        # Change the format of the points
        self.ext_points = [[points[0][i], points[1][i], points[2][i]] for i in range(len(points[0]))]

        # Create the rake in Paraview
        programmableSource = pasi.ProgrammableSource()
        programmableSource.OutputDataSetType = 'vtkPolyData'
        programmableSource.Script = f"""
        import vtk

        # Manually input the external points
        custom_points = {self.ext_points}

        # Create a vtkPoints object to store the points
        points = vtk.vtkPoints()

        # Insert custom points into the vtkPoints object
        for point in custom_points:
            points.InsertNextPoint(point)

        # Create a polyline (a single line connecting the points)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(custom_points))
        for i in range(len(custom_points)):
            lines.InsertCellPoint(i)

        # Create the output PolyData and assign points and lines
        output.SetPoints(points)
        output.SetLines(lines)
        """

        # Pull data from rake
        resample = pasi.ResampleWithDataset()
        resample.SourceDataArrays = [data]
        resample.DestinationMesh = programmableSource
        pasi.UpdatePipeline()

        # Put data in
        self.resampled_output = pasi.servermanager.Fetch(resample)
        point_data = self.resampled_output.GetPointData()
        num_point_arrays = point_data.GetNumberOfArrays()
        self.array_headers = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
        print("Available headers:\t"+str(self.array_headers))

        # Clean up
        pasi.Delete( data )
        pasi.Delete( resample )
        pasi.Delete( programmableSource )
        del data
        del resample
        del programmableSource

        # Restore standard output and error to the default
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Optionally suppress VTK messages entirely
        vtk_output_window = vtk.vtkStringOutputWindow()
        vtk.vtkOutputWindow.SetInstance(vtk_output_window)
        vtk.vtkOutputWindow.GetInstance().SetGlobalWarningDisplay(False)

        self.data_loc = file_format

        # Set coordinate change variable to track if the coordinate change of the rake has occured
        self.coord_change=False
        
    def dataToDictionary( cls ):
        """
        Transfers the data from the Paraview-native format to a Python-native format of a
            dictionary.

        Attributes:

            data {}:    The dictionary containing the data from the rake.

        """

        cls.data = {}

        #
        # Get the coordinates
        #
        points_vtk = cls.resampled_output.GetPoints().GetData()
        points_np = np.asarray( nps.vtk_to_numpy( points_vtk ) )
        for i , c in enumerate( [ 'x' , 'y' , 'z' ] ):
            cls.data[c] = points_np[:,i]

        #
        # Get the data
        #
        for i , d in enumerate( cls.array_headers ):
            data_vtk = cls.resampled_output.GetPointData().GetArray( d )
            data_np = nps.vtk_to_numpy( data_vtk )
            cls.data[d] = data_np

        #del cls.resampled_output

        cls.data_loc = "dictionary"

    def dataToPandas( cls , coords = ['x', 'y', 'z'] ):
        """
        Put the data from the Paraview native format to Pandas. Pandas will be more convenient for
            exporting.

        Args:
            coords (list, optional):    The coordinates for the data. Defaults to ['x', 'y', 'z'].
        """
        

        #
        # Initialize the dataframe with the coordinates
        #
        points_vtk = cls.resampled_output.GetPoints().GetData()
        points_np = nps.vtk_to_numpy( points_vtk )
        cls.data_df = pd.DataFrame( points_np , columns = coords )

        #
        # Put data in dataframe
        #
        for i , d in enumerate( cls.array_headers ):
            data_vtk = cls.resampled_output.GetPointData().GetArray( d )
            data_np = nps.vtk_to_numpy( data_vtk )
            if len( data_np.shape ) > 1 :
                for j , c in enumerate( coords ):
                    data_name = d + c
                    cls.data_df[data_name] = data_np[:,j]
            else:
                cls.data_df[d] = data_np

        cls.data_loc = "pandas"

        #del cls.resampled_output

    def coordinateChange( cls , coord_tol=1e-9 , nDimensions=2 , fix_blanks=False , rot_axis_val=1 ):
        """
        This method takes the data on the rake and transforms it into the coordinate system defined by
            the rake of normal, tangent, 2nd tangent.

        If one wants a better reference for the methods, one should read:

        Advanced Engineering Mathematics, 6th edition: Chapter 9. By Zill.

        """

        if cls.data_loc.lower()=="pandas":
            
            #
            # Calculate the unit vectors
            #
            cls.C = np.asarray( [ cls.data_df["x"].values , cls.data_df["y"].values , cls.data_df["z"].values ] )

            cls.dC = np.gradient( cls.C , axis=1 )
            cls.tangent_vector = cls.dC / np.linalg.norm( cls.dC , axis=0 )
            cls.dtangent_vector = np.gradient( cls.tangent_vector , axis=1 , edge_order=2 )
            """
            for i in range( np.shape( cls.dtangent_vector)[-1] ):
                for j in range(3):
                    if cls.dtangent_vector[j,i]==np.nan:
                        cls.dtangent_vector[j,i]==0

                if np.linalg.norm( cls.dtangent_vector[:,i] )<=coord_tol:
                    if i>0 and i<np.shape( cls.dtangent_vector)[-1]-1:
                        cls.dtangent_vector[:,i] = 0.5 * ( cls.dtangent_vector[:,i-1]+cls.dtangent_vector[:,i+1] )
                    elif i>0:
                        cls.dtangent_vector[:,i] = cls.dtangent_vector[:,i+1]
                    else:
                        cls.dtangent_vector[:,i] = cls.dtangent_vector[:,i-1]
            """

            rotate_axis = np.zeros_like( cls.tangent_vector )
            rotate_axis[-1,:] = rot_axis_val
            if nDimensions==3:
                cls.normal_vector = cls.dtangent_vector / np.linalg.norm( cls.dtangent_vector , axis=0 )
            elif nDimensions==2:
                cls.normal_vector = np.cross( cls.tangent_vector , rotate_axis , axis=0 )
            else:
                raise ValueError("Improper number of dimensions")

            print( "Normal vector is {x:.3f}% efficient".format(x= np.sum( np.linalg.norm( cls.normal_vector , axis=0 ) ) / np.shape( cls.normal_vector )[-1] ) )

            cls.binormal_vector = np.cross( cls.tangent_vector , cls.normal_vector , axis=0 )

            if fix_blanks:
                
                cls.normal_vector[:,np.isnan(cls.normal_vector)]=cls.normal_vector[np.isnan(cls.normal_vector)]
                cls.binormal_vector[np.isnan(cls.binormal_vector)]=0



            #
            # Transform velocity
            #
            cls.U = np.asarray( [ cls.data_df["Ux"].values , cls.data_df["Uy"].values , cls.data_df["Uz"].values ] )

            cls.U_r = np.zeros_like( cls.U )

            for i in range( np.shape( cls.U )[-1] ):

                cls.U_r[1,i] = np.dot( cls.tangent_vector[:,i] , cls.U[:,i] )
                cls.U_r[0,i] = np.dot( cls.normal_vector[:,i] , cls.U[:,i] )
                cls.U_r[2,i] = np.dot( cls.binormal_vector[:,i] , cls.U[:,i] )

            #cls.data_df["C_n"] = cls.normal_vector
            #cls.data_df["C_t"] = cls.tangent_vector
            #cls.data_df["C_b"] = cls.binormal_vector

            cls.data_df["U_n"] = cls.U_r[0,:]
            cls.data_df["U_t"] = cls.U_r[1,:]
            cls.data_df["U_b"] = cls.U_r[2,:]

            cls.C_r = np.zeros_like( cls.C )

            for i in range( np.shape( cls.C )[-1] ):

                cls.C_r[0,i] = np.dot( cls.normal_vector[:,i] , cls.C[:,i] )
                cls.C_r[1,i] = np.dot( cls.tangent_vector[:,i] , cls.C[:,i] )
                cls.C_r[2,i] = np.dot( cls.binormal_vector[:,i] , cls.C[:,i] )

            cls.data_df["C_n"] = cls.C_r[0,:]
            cls.data_df["C_t"] = cls.C_r[1,:]
            cls.data_df["C_b"] = cls.C_r[2,:]

        cls.coord_change=True


    def flowData( cls , nu , side=None , dataDictionaryFormat="pandas" , x_offset=0 ):
        """
        This method provides flow data for the rake, assuming proximity to a wall. 


        Args:
            side (string, optional):    The side that the wall is on. The two available options are:

                                        -"LHS": The side of the rake with the lower index values.

                                        -"RHS": The side of the rake with the higher index values.

                                        The default is None, (TODO:NOPE) which automatically detects the wall. Not
                                            case sensitive.

        Raises:
            ValueError: _description_
        """

        cls.side = side.lower()
        cls.nu = nu

        if not cls.coord_change:

            if side.lower()=="lhs":
                print("Left hand side flow data.")

                if dataDictionaryFormat.lower()=="pandas":
                    print("Pandas data")

                    cls.u = cls.data_df["Ux"].values
                    cls.y = cls.data_df["y"].values
                    #print("Raw y's:\t"+str(cls.y))
                    cls.y[np.abs(cls.y)>0] = cls.y[np.abs(cls.y)>0] * ( cls.y[np.abs(cls.y)>0] / np.abs( cls.y[np.abs(cls.y)>0] ) )
                    #print("Normalized y's:\t"+str(cls.y))
                    cls.x = cls.data_df["x"].values[0] - x_offset
                    cls.delta , cls.delta_star , cls.theta = boundaryLayerThickness( cls.y , cls.u )
                    cls.u_tau , cls.C_f = shearConditions( cls.y , cls.u , nu )
                    #print("u_tau:\t{x:.3f}".format(x=cls.u_tau))
                    cls.Re_x = ReynoldsNumber( cls.x , nu , u = cls.u )
                    cls.Re_delta = ReynoldsNumber( cls.delta , nu , u = cls.u )
                    cls.Re_theta = ReynoldsNumber( cls.theta , nu , u = cls.u )
                    cls.Re_tau = ReynoldsNumber( cls.delta , nu , U_inf=cls.u_tau )
                    cls.delta_x = cls.delta / cls.x
                    cls.delta_star_x = cls.delta_star / cls.x
                    cls.theta_x = cls.theta / cls.x
                    cls.H = cls.delta_star / cls.theta
                
            elif side.lower()=="rhs":
                print("Right hand side flow data.")

                if dataDictionaryFormat.lower()=="pandas":
                    print("Pandas data")

                    cls.u = cls.data_df["Ux"].values[::-1]
                    cls.y = cls.data_df["y"].values[::-1]
                    #print("Raw y's:\t"+str(cls.y))
                    cls.y[np.abs(cls.y)>0] = cls.y[np.abs(cls.y)>0] * ( cls.y[np.abs(cls.y)>0] / np.abs( cls.y[np.abs(cls.y)>0] ) )
                    #print("Normalized y's:\t"+str(cls.y))
                    cls.x = cls.data_df["x"].values[-1] - x_offset
                    cls.delta , cls.delta_star , cls.theta = boundaryLayerThickness( cls.y , cls.u )
                    cls.u_tau , cls.C_f = shearConditions( cls.y , cls.u , nu )
                    #print("u_tau:\t{x:.3f}".format(x=cls.u_tau))
                    cls.Re_x = ReynoldsNumber( cls.x , nu , u = cls.u )
                    cls.Re_delta = ReynoldsNumber( cls.delta , nu , u = cls.u )
                    cls.Re_theta = ReynoldsNumber( cls.theta , nu , u = cls.u )
                    cls.Re_tau = ReynoldsNumber( cls.delta , nu , U_inf=cls.u_tau )
                    cls.delta_x = cls.delta / cls.x
                    cls.delta_star_x = cls.delta_star / cls.x
                    cls.theta_x = cls.theta / cls.x
                    cls.H = cls.delta_star / cls.theta

            elif side==None:
                raise ValueError( "None side not implemented yet" )
            
            else:
                raise ValueError( "Invalid side selected" )
            
        else:

            data_points = len( cls.data_df["U_n"].values )

            if side.lower()=="lhs":
                print("Left hand side flow data.")

                if dataDictionaryFormat.lower()=="pandas":
                    print("Pandas data")

                    cls.u = cls.data_df["U_n"].values[:data_points//2]
                    cls.y = cls.data_df["C_t"].values[:data_points//2] - cls.data_df["C_t"].values[0]
                    #print("Raw y's:\t"+str(cls.y))
                    cls.y[np.abs(cls.y)>0] = cls.y[np.abs(cls.y)>0] * ( cls.y[np.abs(cls.y)>0] / np.abs( cls.y[np.abs(cls.y)>0] ) )
                    #print("Normalized y's:\t"+str(cls.y))
                    cls.x = cls.data_df["C_n"].values[0] - x_offset
                    cls.delta , cls.delta_star , cls.theta = boundaryLayerThickness( cls.y , cls.u )
                    cls.u_tau , cls.C_f = shearConditions( cls.y , cls.u , nu )
                    #print("u_tau:\t{x:.3f}".format(x=cls.u_tau))
                    cls.Re_x = ReynoldsNumber( cls.x , nu , u = cls.u )
                    cls.Re_delta = ReynoldsNumber( cls.delta , nu , u = cls.u )
                    cls.Re_theta = ReynoldsNumber( cls.theta , nu , u = cls.u )
                    cls.Re_tau = ReynoldsNumber( cls.delta , nu , U_inf=cls.u_tau )
                    cls.delta_x = cls.delta / cls.x
                    cls.delta_star_x = cls.delta_star / cls.x
                    cls.theta_x = cls.theta / cls.x
                    cls.H = cls.delta_star / cls.theta
                
            elif side.lower()=="rhs":
                print("Right hand side flow data.")

                if dataDictionaryFormat.lower()=="pandas":
                    print("Pandas data")

                    cls.u = cls.data_df["U_n"].values[:data_points//2:-1]
                    cls.y = np.abs( cls.data_df["C_t"].values[:data_points//2:-1] - cls.data_df["C_t"].values[-1] )
                    #print("Raw y's:\t"+str(cls.y))
                    cls.y[np.abs(cls.y)>0] = cls.y[np.abs(cls.y)>0] * ( cls.y[np.abs(cls.y)>0] / np.abs( cls.y[np.abs(cls.y)>0] ) )
                    #print("Normalized y's:\t"+str(cls.y))
                    cls.x = cls.data_df["C_n"].values[-1] - x_offset
                    cls.delta , cls.delta_star , cls.theta = boundaryLayerThickness( cls.y , cls.u )
                    cls.u_tau , cls.C_f = shearConditions( cls.y , cls.u , nu )
                    #print("u_tau:\t{x:.3f}".format(x=cls.u_tau))
                    cls.Re_x = ReynoldsNumber( cls.x , nu , u = cls.u )
                    cls.Re_delta = ReynoldsNumber( cls.delta , nu , u = cls.u )
                    cls.Re_theta = ReynoldsNumber( cls.theta , nu , u = cls.u )
                    cls.Re_tau = ReynoldsNumber( cls.delta , nu , U_inf=cls.u_tau )
                    cls.delta_x = cls.delta / cls.x
                    cls.delta_star_x = cls.delta_star / cls.x
                    cls.theta_x = cls.theta / cls.x
                    cls.H = cls.delta_star / cls.theta

            elif side==None:
                raise ValueError( "None side not implemented yet" )
            
            else:
                raise ValueError( "Invalid side selected" )
            
    def boundaryLayerProfile( cls , vonKarmanConst=0.41 , C_plus=5.0 , eta_1=11.0 , b=0.33 , profile="gauriniMoser" ):

        cls.yplus = cls.y * cls.u_tau / cls.nu
        cls.uplus = cls.u / cls.u_tau

        if profile.lower()=="simpsonbackflow":

            print("Calculating Simpson backflow profile")

            A=0.3

            cls.u_N = np.min( cls.u )
            cls.N = cls.y[cls.u==cls.u_N]

            cls.u_fit = np.zeros_like( cls.u )

        print("Hello there")

    def closeout( cls ):

        del cls.resampled_output



