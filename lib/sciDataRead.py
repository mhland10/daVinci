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

0.1     2025/02/17  Added the dataReader object to the file so that rake inherits from it.

"""

#==================================================================================================
#
#   Imports
#
#==================================================================================================

import os, glob
import numpy as np
import sys
import pandas as pd
from natsort import natsorted

#==================================================================================================
#
#   Functions for post-processing
#
#==================================================================================================

def point_sweep( anchors, point_Deltas ):
    """
        This function creates a set of points around the anchors based on the point_Deltas.

    Args:
        anchors (numpy ndarray - float):    The anchor points that the sweep will be placed around,
                                                or anchored to, hence the name. The shape must be:

                                                ( 3 (n dimensions), t (n of time/parameter points) )

        point_Deltas (numpy ndarray - float):   The distribution of points around the anchors.

                                                ( 3 (n dimensions), N (n of points at each sweep) )

    Returns:
        points (numpy ndarray - float): The distribution of points at each point in shape:

                                        ( t (number of time/parameter points), 3 (n dimensions), N (n of points at each sweep) )
    
    """

    # Define the shapes of the data
    a_shape = np.shape( anchors )
    d_shape = np.shape( point_Deltas )
    if not a_shape[0]==d_shape[0]:
        raise ValueError( "Anchors and point_Deltas do not have the same number of dimensions associate with the points" )
    p_shape = ( a_shape[1] ,) + ( a_shape[0] ,) + ( d_shape[1] ,)

    # Create and list the 
    points = np.zeros( p_shape )
    for i in range( p_shape[0] ):
        points[i,...] = anchors[:,i][:, np.newaxis] + point_Deltas
    
    return points

#==================================================================================================
#
#   Post-Processing Objects
#
#==================================================================================================

class dataReader:
    """
        This object is a general data reader that allows the user to draw data from the datafiles 
    the object is initialized with. This object will be inherited by the specific data readers. to
    use its functionality.

    """
    def __init__(self, points, datafile, file_format="vtk", t_lims=None):
        """
        Initialize the dataReader object according to the inputs to the file.

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
        
        # Expand wildcard pattern into actual file list with natural sorting
        self.datafile = datafile
        self.file_list = natsorted(glob.glob(datafile))  # Use natsorted instead of sorted
        print("File list:\t"+str(self.file_list))

        # Store the file format
        self.file_format = file_format

        # Store the points on the rake
        self.ext_points = [[points[0][i], points[1][i], points[2][i]] for i in range(len(points[0]))]
        self.points = np.array( self.ext_points )

        # Set coordinate change variable to track if the coordinate change has occured
        self.coord_change=False   
        self.time_dependent=False

        # Set the time limits
        self.t_lims = t_lims

    def foamCaseRead( cls, working_dir, file_name="foam.foam", verbosity=0, vector_headers=["U"], coordinate_system=['x', 'y', 'z'], interpolator="rbf", accelerator=None, headers_read=None ):
        """
            This reader reads an OpenFOAM case using Paraview

        Args:
            working_dir (string):   The directory of the case.

            file_name (string, optional):   The name of the file to read the OpenFOAM data.
                                                Defaults to "foam.foam"

            verbosity (int, optional):  The verbosity of the reader. 0 is no output, 1 is some 
                                        output, and 2 is all the output.

            vector_headers (list, optional):    The list of headers that are vectors that need to 
                                                be split into their respective components. Defaults
                                                to ["U"].

            coordinate_system (list, optional):    The coordinate system that will be used. Applies
                                                    to the vector headers and the interpolation. 
                                                    Defaults to ['x', 'y', 'z'].

            interpolator (string, optional):    The interpolator that will be used. The valid 
                                                options are:

                                                - *"rbf", "rbfinterpolator", 
                                                    "radial basis function", "radialbasisfunction":

                                                    For the radial basis function interpolator.

                                                - "l", "lin", "linear", "linearnd", "delaunay", 
                                                    "delaunaytriangulation":

                                                    For the linear interpolator. Uses Delaunay
                                                    triangulation on the back end.

                                                Not cases sensitive.

            accelerator (string, optional):    The accelerator that will be used. The valid options are:

                                                - *None:   No accelerator will be used.

                                                - "cuda", "cu", "c", "cupy":   The CUDA accelerator
                                                    will be used via CuPy.

                                                Not case sensitive.                                            

        """
        # Move to the working directory
        os.chdir( working_dir )

        # Find the number of dimension from the coordinate system
        N_dims = len( coordinate_system )

        #
        # Import modules
        #
        import paraview.simple as pasi
        import paraview.servermanager as pase
        if not accelerator:
            import scipy.interpolate as sint
        elif accelerator.lower() in ["cuda", "cu", "c", "cupy"]:
            import cupyx.scipy.interpolate as sint
        else:
            raise ValueError( "Invalid accelerator engine selected" )

        # Read the OpenFOAM file
        full_flnm = working_dir+"\\"+file_name
        if verbosity>0:
            print(f"The full filename is {full_flnm}")
        cls.foam = pasi.OpenFOAMReader( registrationName=file_name, FileName=full_flnm )
        cls.foam.MeshRegions = ["internalMesh"]
        cls.foam.UpdatePipeline()

        # Check if cells exist now
        if verbosity>0:
            print("Number of cells:", cls.foam.GetDataInformation().GetNumberOfCells())
        
        # Pull the time steps available
        cls.time_steps = cls.foam.TimestepValues

        # Get cell centers
        cls.cell_centers = pasi.CellCenters(Input=cls.foam)
        cls.cell_centers.UpdatePipeline()

        # Fetch data into a numpy-compatible format
        cell_data = pase.Fetch(cls.foam)
        centers_data = pase.Fetch(cls.cell_centers)
        if verbosity>0:
            print(f"Number of blocks in cell centers: {centers_data.GetNumberOfBlocks()}")
            print(f"Number of blocks in cell data: {cell_data.GetNumberOfBlocks()}")

        # Extract the internal mesh block
        internal_centers = centers_data.GetBlock(0)
        internal_cells = cell_data.GetBlock(0)
        if internal_centers is None or internal_cells is None:
            raise ValueError("No valid internal mesh block found in the OpenFOAM case.")
        
        # Initialize storage for extracted data
        cls.data = {internal_cells.GetCellData().GetArrayName(i): [] for i in range( internal_cells.GetCellData().GetNumberOfArrays() ) if internal_cells.GetCellData().GetArrayName(i) not in vector_headers }
        for h in vector_headers:
            for c in coordinate_system:
                cls.data[h+":"+c] = []

        for j, t in enumerate( cls.time_steps ):
            print(f"Current time step {t} at index {j}...")

            # Move the Paraview reader along
            cls.foam.UpdatePipeline(t)

            # Get cell centers
            cls.cell_centers = pasi.CellCenters(Input=cls.foam)
            cls.cell_centers.UpdatePipeline()

            # Fetch data into a numpy-compatible format
            cell_data = pase.Fetch(cls.foam)
            centers_data = pase.Fetch(cls.cell_centers)
            if verbosity>1:
                print(f"Number of blocks in cell centers: {centers_data.GetNumberOfBlocks()}")
                print(f"Number of blocks in cell data: {cell_data.GetNumberOfBlocks()}")

            # Extract the internal mesh block
            internal_centers = centers_data.GetBlock(0)
            internal_cells = cell_data.GetBlock(0)
            if internal_centers is None or internal_cells is None:
                raise ValueError("No valid internal mesh block found in the OpenFOAM case.")

            # Extract cell center coordinates
            num_cells = internal_centers.GetNumberOfPoints()
            cell_coords = np.array([internal_centers.GetPoint(i) for i in range(num_cells)])
            cls.cell_coords = cell_coords

            # Extract field data
            num_arrays = internal_cells.GetCellData().GetNumberOfArrays()
            data_dict = {}
            for i in range(num_arrays):
                name = internal_cells.GetCellData().GetArrayName(i)
                if not headers_read:
                    array = internal_cells.GetCellData().GetArray(i)
                    data_dict[name] = np.array([array.GetTuple(i) for i in range(num_cells)])
                else:
                    if name in vector_headers:
                        array = internal_cells.GetCellData().GetArray(i)
                        data_dict[name] = np.array([array.GetTuple(i) for i in range(num_cells)])
                    if name in headers_read:
                        array = internal_cells.GetCellData().GetArray(i)
                        data_dict[name] = np.array([array.GetTuple(i) for i in range(num_cells)])
            for h in vector_headers:
                for i in range( data_dict[h].shape[-1] ):
                    if i<N_dims:
                        data_dict[h+":"+coordinate_system[i]]=np.array([list(data_dict[h][:,i])]).T
                data_dict.pop(h)
            cls.data_dict = data_dict
            if verbosity>1:
                for k in list( data_dict.keys() ):
                    print( f"Key {k} has shape {data_dict[k].shape}" )
            data_matrix = np.array([ data_dict[k][...,0] for k in list( data_dict.keys() ) ])
            cls.data_matrix = data_matrix

            # Print summary
            if verbosity>1:

                # Combine coordinates and field data
                full_data = {"coordinates": cell_coords}
                full_data.update(data_dict)

                print(f"Extracted {len(cell_coords)} cell centers.")
                print("Available fields:", list(full_data.keys()))
                print("First 5 cell centers:\n", cell_coords[:5])
                if "U" in full_data:
                    print("First 5 velocity vectors:\n", full_data["U"][:5])

            # Interpolate onto the points
            if interpolator.lower() in ["rbf", "rbfinterpolator","radial basis function", "radialbasisfunction"]:
                if 0.0 in np.sum( cell_coords[:,:N_dims] , axis=0 ):
                    #print("Actually, it's 1D")
                    drop_dim = np.argmin( np.abs( np.sum( cell_coords[:,:N_dims] , axis=0 ) ) )
                    print(f"Dropping dimension {drop_dim}")
                    object_data = sint.RBFInterpolator( np.delete( cell_coords[:,:N_dims], drop_dim, axis=1 ), data_matrix.T )( np.delete( cls.points[:,:N_dims], drop_dim, axis=1 ) )
                else:
                    object_data = sint.RBFInterpolator( cell_coords[:,:N_dims], data_matrix.T )( cls.points[:,:N_dims] )
            elif interpolator.lower() in ["l", "lin", "linear", "linearnd", "delaunay", "delaunaytriangulation"]:
                if 0.0 in np.sum( cell_coords[:,:N_dims] , axis=0 ):
                    #print("Actually, it's 1D")
                    drop_dim = np.argmin( np.abs( np.sum( cell_coords[:,:N_dims] , axis=0 ) ) )
                    print(f"Dropping dimension {drop_dim}")
                    x = np.delete( cell_coords[:,:N_dims], drop_dim, axis=1 ).reshape( np.shape(np.delete( cell_coords[:,:N_dims], drop_dim, axis=1 ))[0] )
                    y = data_matrix.T
                    x_new = np.delete( cls.points[:,:N_dims], drop_dim, axis=1 ).reshape( np.shape(np.delete( cls.points[:,:N_dims], drop_dim, axis=1 ))[0] )
                    print(f"x is shape {np.shape(x)} and y is shape {np.shape(y)}")
                    print(f"x_new is shape {np.shape(x_new)} in [{np.min(x_new)}, {np.max(x_new)}]")
                    object_data = sint.interp1d( x , y , kind="linear", axis=0 )( x_new )
                else:
                    object_data = sint.LinearNDInterpolator( cell_coords[:,:N_dims ], data_matrix.T )( cls.points[:,:N_dims] )
            else:
                raise ValueError( "Invalid interpolator selected" )
            
            # Re-arrange back into the keys
            for i, k in enumerate( list( data_dict.keys() ) ):
                if not accelerator:
                    cls.data[k] += [object_data[:,i]]
                elif accelerator.lower() in ["cuda", "cu", "c", "cupy"]:
                    cls.data[k] += [object_data[:,i].get()]

        if not cls.time_dependent:
            for i, k in enumerate( list( cls.data.keys() ) ):
                
                # Check if there is a homogeneous size
                try:
                    cls.data[k] = np.array( cls.data[k] )
                except:
                    raise Warning( "The data is not truly time dependent" )

    def paraviewDataRead( cls , working_dir , trim_headers=["vtkValidPointMask"] , coords = ["X","Y","Z"], data="point" ):
        """
        This method reads the data using the Paraview engine and stores the data in the rake object
            in a dictionary.

        Args:
        
            working_dir (string):   The path to the working directory write/read the temporary
                                        files used to get the data in/out.

            trim_headers (list, optional):  The headers that will be trimmed from the data. Defaults
                                                to ["vtkValidPointMask"].

            coords (list, optional):    The coordinates that will be used in the data. Defaults to
                                            ["X","Y","Z"].

        """

        import paraview.simple as pasi
        import vtk

        os.chdir( working_dir )

        # If the file format is inferred, find what it is
        if cls.file_format is None:
            ext = os.path.splitext(datafile)[-1].lower()
            if ext == ".vtk":
                print("VTK file format detected.")
                cls.file_format = "vtk"
            elif ext == ".h5":
                print("H5 file format detected.")
                cls.file_format = "h5"
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
        
        # Load the data from the *.vtk files
        if not cls.file_list:
            raise FileNotFoundError(f"No files found matching {cls.datafile}")
        if cls.file_format.lower()=="vtk" or cls.file_format.lower()==".vtk":
            data = pasi.OpenDataFile( cls.file_list )
        elif cls.file_format.lower()=="h5" or cls.file_format.lower()==".h5":
            data = pasi.CONVERGECFDReader(FileName=cls.file_list[0])
        else:
            raise ValueError(f"Unsupported file format:\t{cls.file_format}")
        #print("Available data attributes:\t"+str(dir(data)))

        # Get available time steps
        cls.time_steps = data.TimestepValues
        print(f"Time steps available: {cls.time_steps}")

        # Create the rake in Paraview
        programmableSource = pasi.ProgrammableSource()
        programmableSource.OutputDataSetType = 'vtkPolyData'
        programmableSource.Script = f"""
        import vtk

        # Manually input the external points
        custom_points = {cls.ext_points}

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

        #
        #   Get point data for initialization
        # 
        resample = pasi.ResampleWithDataset()
        resample.SourceDataArrays = [data]
        resample.DestinationMesh = programmableSource
        #print("Resample attributes:\t"+str(dir(resample)))
        pasi.UpdatePipeline()
        resampled_output = pasi.servermanager.Fetch(resample)
        if data.lower() in ["p", "point", "points", "pointdata"]:
            point_data = resampled_output.GetPointData()
        elif data.lower() in ["c", "cell", "cells", "celldata"]:
            point_data = resampled_output.GetCellData()

        # Get array headers
        cls.array_headers = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
        print("Available headers:\t" + str(cls.array_headers))

        # Initialize storage for extracted data
        cls.data_dict = {header: [] for header in cls.array_headers}

        # Iterate through time steps
        # TODO: Make parallel
        for t in cls.time_steps:
            data.UpdatePipeline(time=t)  # Ensure the data is updated for this time step
            
            # Resample at current time step
            resample = pasi.ResampleWithDataset()
            resample.SourceDataArrays = [data]
            resample.DestinationMesh = programmableSource
            #resample.InterpolationType = "Linear"  # or "Cubic" depending on the method available
            pasi.UpdatePipeline()

            # Fetch resampled output
            resampled_output = pasi.servermanager.Fetch(resample)
            if data.lower() in ["p", "point", "points", "pointdata"]:
                point_data = resampled_output.GetPointData()
            elif data.lower() in ["c", "cell", "cells", "celldata"]:
                point_data = resampled_output.GetCellData()


            # Extract data for each variable and store it
            for header in cls.array_headers:
                array = point_data.GetArray(header)
                if array:
                    print(f"Array {header} has {array.GetNumberOfComponents()} components and {array.GetNumberOfTuples()} tuples")
                    # Check if the array is a vector (e.g., velocity)
                    if array.GetNumberOfComponents() == 3:
                        for i in range(3):
                            component_header = f"{header}:{coords[i]}"
                            # Initialize the key in cls.data_dict if it doesn't exist
                            if component_header not in cls.data_dict:
                                cls.data_dict[component_header] = []
                            cls.data_dict[component_header].append([array.GetComponent(j, i) for j in range(array.GetNumberOfTuples())])
                    else:
                        cls.data_dict[header].append([array.GetValue(i) for i in range(array.GetNumberOfTuples())])

            #for header in cls.array_headers:
            #    array = point_data.GetArray(header)
            #    if array:
            #        cls.data_dict[header].append([array.GetValue(i) for i in range(array.GetNumberOfTuples())])

            # Clean up for memory efficiency
            pasi.Delete(resample)
            del resample

        # Convert lists to NumPy arrays to have shape (num_timesteps, num_rake_points)
        for header in cls.data_dict:
            cls.data_dict[header] = np.array(cls.data_dict[header])  

        # Clean up
        pasi.Delete( data )
        pasi.Delete( programmableSource )
        del data
        del programmableSource

        # Restore standard output and error to the default
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Optionally suppress VTK messages entirely
        vtk_output_window = vtk.vtkStringOutputWindow()
        vtk.vtkOutputWindow.SetInstance(vtk_output_window)
        vtk.vtkOutputWindow.GetInstance().SetGlobalWarningDisplay(False) 

        # Trim and store data
        for p in trim_headers:
            if p in list(cls.data_dict.keys()):
                cls.data_dict.pop(p)
        cls.data = cls.data_dict
        
    def convergeH5DataRead( cls , working_dir , data_prefix="data_ts" , sig_figs=6 , N_dims=3 , interpolator="RBF" , overwrite=False , write=True , rm_after_read=False , mp_method=None , N_cores=None, accelerator=None ):
        """
        This method reads the data using the Converge engine and stores the data in the rake object
            in a dictionary.

        Args:

            working_dir (string):   The path to the working directory write/read the temporary
                                        files used to get the data in/out.

            data_prefix (string, optional):  The prefix of the *.csv files used to temporarily 
                                                write the data.

            sig_figs (int, optional):   How many significant figures will be used in the temporary
                                            files to translate the data.

            N_dims (int, optional): How many dimensions will be considered in the interpolation.

            interpolator (string, optional):    Which interpolating function will be used. The
                                                    valid options are, not case sensitive:

                                                - *"RBF":   The SciPy RBF (radial basis function)
                                                                interpolator.

                                                - "CT":     The SciPy CloughTocher2D interpolator.
                                                                Note that this only works when
                                                                N_dim=2.

                                                - "Linear" or "Lin":    The SciPy linear 
                                                                        interpolator. Uses Delaunay
                                                                        triangulation.

            overwrite (boolean, optional):  Whether the reader will overwrite *.csv files if they
                                                are already present. Defaults to False.

            write (boolean, optional):  Whether the reader will overwrite any file automatically.
                                            Defaults to True.
            
            rm_after_read (boolean, optional):  Whether the *.csv file that corresponds to a time
                                                    step will be deleted after reading the data
                                                    into a Pandas dataframe. Defaults to False.

            mp_method (string, optional):   Which multiprocessing method to parallelize the
                                                operations. Not currently implemented.

            N_cores (string, optional): The number of processes that multiprocessing can use.

        """

        import paraview.simple as pasi
        import pandas as pd
        if not accelerator:
            import scipy.interpolate as sint
        elif accelerator.lower() in ["cuda", "cu", "c", "cupy"]:
            import cupyx.scipy.interpolate as sint
        else:
            raise ValueError( "Invalid accelerator engine selected" )
        
        # Load the data from the files
        cls.data = pasi.CONVERGECFDReader(FileName=cls.file_list[0])
        cls.data.SMProxy.SetAnnotation("ParaView::Name", "MyData")
        print("Available data attributes:\t"+str(dir(cls.data)))

        # Get available time steps
        cls.time_steps = cls.data.TimestepValues
        print(f"Time steps available: {cls.time_steps}")

        # Get original directory
        og_dir = os.getcwd()

        # Set up data dictionary
        cls.data = {}
        if cls.time_dependent:
            shape_matrix = np.zeros( ( len(cls.time_steps) , np.shape( cls.points )[1] ) )
        else:
            shape_matrix = np.zeros( ( len(cls.time_steps) , np.shape( cls.ext_points )[0] ) )

        # Find Source
        sources = pasi.GetSources()
        print(f"Available sources: {sources}")
        if sources:
            source_name = list(sources.keys())[0][0]  # Extract the string name
            source = pasi.FindSource(source_name)
            print(f"Using source: {source_name}")
        else:
            print("No sources found!")
        cls.source = source

        # Extract Coordinates
        print("Appending Coordinates...")
        cls.coordinates1 = pasi.Coordinates(registrationName='Coordinates1', Input=source)
        cls.coordinates1.AppendPointLocations = 0
        cls.coordinates1.AppendCellCenters = 1

        # Writing data to csv
        os.chdir( working_dir )
        file_write_nm = working_dir + "\\" + data_prefix + "_" + str(len(cls.time_steps)-1) + ".csv"
        file_start = working_dir + "\\" + data_prefix + ".csv"
        if (overwrite or not os.path.exists(file_write_nm)) and write:
            print(f"Changing directory to {os.getcwd()}\n to write {file_write_nm}")
            pasi.SaveData( file_start , proxy=cls.coordinates1, 
                                    WriteTimeSteps=1, WriteTimeStepsSeparately=1, 
                                    Filenamesuffix='_%d', ChooseArraysToWrite=0, 
                                    PointDataArrays=[], 
                                    CellDataArrays = cls.source.CellData.keys() + ["CellCenters"] , 
                                    FieldDataArrays=[], VertexDataArrays=[], EdgeDataArrays=[], 
                                    RowDataArrays=[], Precision=sig_figs, UseStringDelimiter=1, 
                                    UseScientificNotation=1, FieldAssociation='Cell Data', 
                                    AddMetaData=0, AddTimeStep=0, AddTime=0)

        if not mp_method:

            for t_i , t in enumerate( cls.time_steps ):

                print(f"Time Index:\t{t_i}")        
                file_write_nm = working_dir + "\\" + data_prefix + "_" + str(t_i) + ".csv"            
                
                # Pull data into dataframe
                print(f"Reading {file_write_nm}...")
                cls.df_read = pd.read_csv( file_write_nm,
                                    sep=',',  # Use '\t' if your file is tab-delimited
                                    header=0,  # The first row as column names
                                    )
                if rm_after_read:
                    os.remove(file_write_nm)
                
                # Separate into coordinates and data
                data_columns = [col for col in cls.df_read.columns if not col.startswith('CellCenters')]
                cls.df_data = cls.df_read[data_columns]
                coord_columns = [col for col in cls.df_read.columns if col.startswith('CellCenters')]
                cls.df_coord = cls.df_read[coord_columns]

                # Set up the coordinates into numpy array/matrices
                cls.coordinates = np.zeros( ( len( cls.df_coord[ cls.df_coord.keys()[0] ].to_numpy() ) , N_dims ) )
                for i in range( N_dims ):
                    cls.coordinates[:,i] = cls.df_coord[ cls.df_coord.keys()[i] ].to_numpy()
                if cls.time_dependent:
                    cls.rake_coordinates = cls.points[t_i,:,:N_dims]
                else:
                    cls.rake_coordinates = np.zeros( ( np.shape( cls.ext_points )[0] , N_dims ) )
                    for i in range( np.shape( cls.ext_points )[0] ):
                        cls.rake_coordinates[i,:] = cls.ext_points[i][:N_dims]

                # Initialize the dictionary
                if t_i==0:
                    cls.data_raw = np.zeros( np.shape( shape_matrix ) + ( len(cls.df_data.columns) ,) )
                    for i , h in enumerate( cls.df_data.columns ):
                        cls.data[h] = np.zeros_like( shape_matrix )

                # Break the data into a dictionary
                # TODO: RBFInterpolator() takes up too much RAM. At some point, we should make an optimized
                #           version that can fit into limited RAM.
                print("Interpolating points...")
                """
                for i , h in enumerate( cls.df_data.columns ):
                    col_data = cls.df_data[h].to_numpy()
                    if interpolator.lower()=="rbf":
                        cls.data[h][t_i,:] = sint.RBFInterpolator( cls.coordinates , col_data )( cls.rake_coordinates )
                    elif interpolator.lower()=="ct" and N_dims==2:
                        cls.data[h][t_i,:] = sint.CloughTocher2DInterpolator( cls.coordinates , col_data )( cls.rake_coordinates )
                    elif interpolator.lower()=="linear" or interpolator.lower()=="lin":
                        cls.data[h][t_i,:] = sint.LinearNDInterpolator( cls.coordinates , col_data )( cls.rake_coordinates )
                    else:
                        raise ValueError("Invalid interpolator selected.")
                #"""
                #"""
                col_data = []
                for i , h in enumerate( cls.df_data.columns ):
                    col_data += [cls.df_data[h].to_numpy()]
                col_data = np.array( col_data ).T

                print("Coordinates are shape:\t",np.shape(cls.coordinates))
                print("Data is shape:\t",np.shape(col_data))
                print("Sampling coordinates are shape:\t",np.shape(cls.rake_coordinates))
                print("Column data sample:\t",col_data)

                if interpolator.lower()=="rbf":
                    print("Interpolating with RBF interpolator...")
                    cls.data_raw[t_i,...] = sint.RBFInterpolator( cls.coordinates , col_data )( cls.rake_coordinates )
                elif interpolator.lower()=="ct" and N_dims==2:
                    print("Interpolating with Clough Tocher interpolator...")
                    cls.data_raw[t_i,...] = sint.CloughTocher2DInterpolator( cls.coordinates , col_data )( cls.rake_coordinates )
                elif interpolator.lower()=="linear" or interpolator.lower()=="lin":
                    print("Interpolating with linear interpolator...")
                    cls.data_raw[t_i,...] = sint.LinearNDInterpolator( cls.coordinates , col_data )( cls.rake_coordinates )
                else:
                    raise ValueError("Invalid interpolator selected.")
                print("Interpolation finished")
                
                for i , h in enumerate( cls.df_data.columns ):
                    print(f"Putting data back in for key {h}")
                    if not accelerator:
                        cls.data[h][t_i,:] = cls.data_raw[t_i,:,i]
                    elif accelerator.lower() in ["cuda", "cu", "c", "cupy"]:
                        cls.data[h][t_i,:] = cls.data_raw[t_i,:,i].get()
                #"""

                print(f"Interpolation Finished for Time Index {t_i}")

        elif mp_method.lower()=="mpi":

            # Set import MPI
            print("MPI multiprocessing method is under construction.")
            from mpi4py import MPI
            comm=MPI.COMM_WORLD
            rank=comm.Get_rank()
            size=comm.Get_size()

            # Split the time steps
            steps_xRank = len( cls.time_steps )//size
            remainders = len( cls.time_steps )%size
            start_xRank = rank * steps_xRank + min(rank, remainders)
            end_xRank = start_xRank + steps_xRank + (1 if rank < remainders else 0)

            # Set up the data dictionary
            if rank==0:
                for i , h in enumerate( cls.df_data.columns ):
                        cls.data[h] = np.zeros_like( shape_matrix )

            # Pull the data from the intermediate *.csvs into the dictionary
            for t_i_raw , t in enumerate( cls.time_steps[start_xRank[rank]:end_xRank[rank]] ):
                t_i = t_i_raw + start_xRank[rank]

                print(f"Rank {rank} is reading:\t")
                print(f"Time Index:\t{t_i}")        
                file_write_nm = working_dir + "\\" + data_prefix + "_" + str(t_i) + ".csv"            
                
                # Pull data into dataframe
                print(f"Reading {file_write_nm}...")
                df_read = pd.read_csv( file_write_nm,
                                    sep=',',  # Use '\t' if your file is tab-delimited
                                    header=0,  # The first row as column names
                                    )
                if rm_after_read:
                    os.remove(file_write_nm)
                
                # Separate into coordinates and data
                data_columns = [col for col in df_read.columns if not col.startswith('CellCenters')]
                df_data = df_read[data_columns]
                coord_columns = [col for col in df_read.columns if col.startswith('CellCenters')]
                df_coord = df_read[coord_columns]

                # Set up the coordinates into numpy array/matrices
                coordinates = np.zeros( ( len( df_coord[ cls.df_coord.keys()[0] ].to_numpy() ) , N_dims ) )
                for i in range( N_dims ):
                    coordinates[:,i] = df_coord[ df_coord.keys()[i] ].to_numpy()
                if cls.time_dependent:
                    rake_coordinates = cls.points[t_i,:,:N_dims]
                else:
                    rake_coordinates = np.zeros( ( np.shape( cls.ext_points )[0] , N_dims ) )
                    for i in range( np.shape( cls.ext_points )[0] ):
                        rake_coordinates[i,:] = cls.ext_points[i][:N_dims]

                # Break the data into a dictionary
                # TODO: RBFInterpolator() takes up too much RAM. At some point, we should make an optimized
                #           version that can fit into limited RAM.
                print("Interpolating points...")
                for i , h in enumerate( df_data.columns ):
                    col_data = df_data[h].to_numpy()
                    if interpolator.lower()=="rbf":
                        cls.data[h][t_i,:] = sint.RBFInterpolator( coordinates , col_data )( rake_coordinates )
                    elif interpolator.lower()=="ct" and N_dims==2:
                        cls.data[h][t_i,:] = sint.CloughTocher2DInterpolator( coordinates , col_data )( rake_coordinates )
                    elif interpolator.lower()=="linear" or interpolator.lower()=="lin":
                        cls.data[h][t_i,:] = sint.LinearNDInterpolator( coordinates , col_data )( rake_coordinates )
                    else:
                        raise ValueError("Invalid interpolator selected.")
                print(f"Interpolation Finished for Time Index {t_i}")

                                
        else:
            raise ValueError("No valid multiprocessing method selected.")
                
        # Return to original directory
        os.chdir(og_dir)

    def hdf5DataRead(cls, working_dir, group_path=["STREAM_00","CELL_CENTER_DATA"], coord_prefix="XCEN", dims=['x','y','z'], interpolator="lin", coords_system=['x','y','z'], mp_method=None ):
        """
            This method takes the data from a *.h5 file or such and imports it using h5py rather 
        than Paraview, which is very slow.

            Note that the dimensions must match the dimension in the CFD analysis, not the data
        object.

        Args:
            working_dir (string):   The directory to work the data from.

            group_path (list, optional):    This list of groups to call, in order to reach data one
                                                is looking for. Defaults to 
                                                ["STREAM_00","CELL_CENTER_DATA"], which is the 
                                                default for Converge's h5 files for the cell-
                                                centered data. Note that this is case sensitive.

            coord_prefix (string, optional):    The string that defines the prefix for the keys
                                                    that define the coordinates. Case sensitive. 
                                                    Defaults to "XCEN", which is the default for
                                                    Converge's h5 files if the cell centers are
                                                    exported.
                                    
            dims (char, optional):  The list of characters that define the dimensions. Defaults to
                                        ['x', 'y', 'z']. Case sensitive.

            interpolator (string, optional):    Which interpolating function will be used. The
                                                    valid options are, not case sensitive:

                                                - *"RBF":   The SciPy RBF (radial basis function)
                                                                interpolator.

                                                - "CT":     The SciPy CloughTocher2D interpolator.
                                                                Note that this only works when
                                                                N_dim=2.

                                                - "Linear" or "Lin":    The SciPy linear 
                                                                        interpolator. Uses Delaunay
                                                                        triangulation.


        """

        import h5py as h5
        import scipy.interpolate as sint

        # Set number of dimension
        N_dims = len( dims )

        
        #
        # Import time step from file format
        #
        cls.time_steps = np.zeros(len(cls.file_list))
        print(f"Seeing {len(cls.time_steps)} time steps.")
        if cls.file_format.lower()=="h5":
            for i in range( len( cls.file_list ) ):
                fl_nm = cls.file_list[i]
                raw_name = fl_nm[:-3]
                time_value = raw_name.split('_')[1][1:]
                cls.time_steps[i] = float( time_value )
                #print(f"Time value at i={i}:\t{time_value}")
        else:
            raise ValueError("Invalid file format for the method requested.")
        # Filter time steps as needed
        if not cls.t_lims is None:
            filtered_inidices = [i for i in range(len(cls.time_steps)) if cls.time_steps[i]>=np.min(cls.t_lims) and cls.time_steps[i]<=np.max(cls.t_lims)]
            print(f"Filtered indices:\t{filtered_inidices}")
            time_steps_filt = [cls.time_steps[i] for i in filtered_inidices]
            cls.file_list = [cls.file_list[i] for i in filtered_inidices]
            cls.time_steps = time_steps_filt
            print(f"Time steps filtered to:\t{cls.time_steps}")
            print(f"Which is {len(cls.time_steps)} time steps.")
        
        #
        # Set up the coordinates into numpy array/matrices, if not time dependent
        #
        if not cls.time_dependent:
            raw_coordinates = np.array( cls.ext_points )
            obj_coordinates = np.zeros_like( raw_coordinates[:,:N_dims])
            for i in range( N_dims ):
                c = dims[i]
                for j in range( len(coords_system) ):
                    cc = coords_system[j]
                    if c.lower()==cc.lower():
                        #print(f"Coordinate {c} matches {cc}.")
                        obj_coordinates[:,i] = raw_coordinates[:,j] 
            #print(f"The object coordinates are in shape:\t{np.shape(obj_coordinates)}")
            cls.obj_coordinates = obj_coordinates
            data_shape = ( len(cls.time_steps) ,) + ( len( obj_coordinates[:,0] ) ,)
        else:
            data_shape = ( len(cls.time_steps) ,) + ( len( cls.points[0] ) ,)

        # Initialize our data
        data_file_path = "/".join(group_path)
        with h5.File( cls.file_list[1], 'r') as f:
            group = f[data_file_path]
            keys = list(group.keys())
            data_keys = [key for key in keys if not key.startswith(coord_prefix)]
        data_array = np.zeros( data_shape + ( len(data_keys) ,) )
        #print(f"Data array shape is {np.shape(data_array)}")
        cls.data={}
        for k in data_keys:
            cls.data[k] = np.zeros( data_shape )

        #
        # Go through the time steps and interpolate the data
        #
        if not mp_method:
            for i in range( len(cls.time_steps) ):
                print(f"Working on data index:\t{i}")
                fl_nm = cls.file_list[i]
                print(f"Working with file:\t{fl_nm}")
                t_step = cls.time_steps[i]


                #
                # Set up the coordinates into numpy array/matrices, if time dependent
                #
                if cls.time_dependent:
                    obj_coordinates = cls.points[i,:,:N_dims]
                    cls.obj_coordinates = obj_coordinates
                    """
                    # Initialize our data
                    data_file_path = "/".join(group_path)
                    with h5.File( cls.file_list[1], 'r') as f:
                        group = f[data_file_path]
                        keys = list(group.keys())
                        data_keys = [key for key in keys if not key.startswith(coord_prefix)]
                    data_array = np.zeros( ( len(cls.time_steps) ,) + ( len( obj_coordinates[:,0] ) ,) + ( len(data_keys) ,) )
                    #print(f"Data array shape is {np.shape(data_array)}")
                    cls.data={}
                    for k in data_keys:
                        cls.data[k] = np.zeros( ( len(cls.time_steps) ,) + ( len( obj_coordinates[:,0] ) ,) )
                    #"""

                #
                # Open the h5 file and deposit data
                #
                with h5.File( fl_nm, 'r') as f:

                    # Pull the data into the group
                    group = f[data_file_path]

                    # Get the keys within the group
                    keys = list(group.keys())
                    #print(f"Keys available:\t{keys}")

                    # Pull the cell center data
                    coord_keys_raw = [key for key in keys if key.startswith("XCEN")]
                    coord_keys = []
                    for c in coord_keys_raw:
                        if c[-1].lower() in dims:
                            coord_keys += [c]
                    #print(f"The coordinate keys are:\t{coord_keys}")
                    coordinates = np.zeros( (N_dims ,) + ( len( group[coord_keys[0]] ) ,) ).T
                    #print(f"Coordinates have shape:\t{np.shape(coordinates)}")
                    for j in range( N_dims ):
                        coordinates[:,j] = group[coord_keys[j]][:]
                    cls.coordinates = coordinates

                    # Store the data into an array for interpolation
                    data_keys = [key for key in keys if not key.startswith(coord_prefix)]
                    #print(f"The data keys are:\t{data_keys}")
                    data_raw = np.zeros( ( len( group[coord_keys[0]] ) ,) + ( len(data_keys) ,) )
                    #print(f"Raw data shape:\t{np.shape(data_raw)}")
                    for j in range( len( data_keys ) ):
                        data_raw[:,j] = group[data_keys[j]][:]
                print("**Data pull complete**")

                #
                # Do the interpolation
                #
                #print(f"Coordinates has shape:\t{np.shape(coordinates)}")
                cls.coordinates = coordinates
                #print(f"Data raw has shape:\t{np.shape(data_raw)}")
                #print(f"Object coordinates has shape:\t{np.shape(obj_coordinates)}")
                if interpolator.lower() in ["lin","linear"]:
                    data_interpolator_raw = sint.LinearNDInterpolator( coordinates, data_raw )( obj_coordinates )
                elif interpolator.lower() in ["rbf","radial basis function"]:
                    data_interpolator_raw = sint.RBFInterpolator( coordinates, data_raw )( obj_coordinates )
                else:
                    raise ValueError("Invalid interpolator selected.")
                #print(f"Original data:\t{data_raw}")
                #print(f"New data:\t{data_interpolator_raw}")
                data_array[i,...] = data_interpolator_raw
                print("**Interpolation complete**")

                #
                # Move data into dictionary
                #
                for j in range( len( data_keys ) ):
                    k = data_keys[j]
                    #print(f"Depositing data for keys {k}:")
                    #print(f"\t{data_array[i,:,j]}")
                    cls.data[k][i,:] = data_array[i,:,j]
                    #print(f"\t{cls.data[k]}")
                    #print(f"\tMin Data:\t{np.min(cls.data[k][i,...])}")
                    #print(f"\tMax Data:\t{np.max(cls.data[k][i,...])}")
                    #print(f"\tMean Data:\t{np.mean(cls.data[k][i,...])}")

        else:
            raise ValueError("Invalid multiprocessing method chosen.")
        

    def hdf5Write( cls , filename , working_dir ):
        """
        This method writes cls.data to an *.h5 file.

        Args:
            filename (string):      The file name.
            working_dir (string):   The directory where the file is.

        """

        import h5py as h5

        with h5.File(os.path.join(working_dir, filename + ".h5"), "a") as f:
            for key, value in cls.data.items():
                if key in f:
                    del f[key]  # Remove existing dataset
                f.create_dataset(key, data=value)

            if "timesteps" in f:
                del f["timesteps"]
            f.create_dataset("timesteps", data=cls.time_steps)

    def hdf5Read( cls , filename , working_dir ):
        """
        This method reads the data from an *.h5 file that corresponds to the rake object.

        Args:
            filename (string):      The file name.
            working_dir (string):   The directory where the file is.

        """

        import h5py as h5

        cls.data = {}
        with h5.File(os.path.join(working_dir, filename + ".h5"), "r") as f:
            # Get all the keys
            keys_avail = list(f.keys())
            print(f"Available keys:\t{keys_avail}")

            for k in keys_avail:
                if k=="timesteps":
                    cls.time_steps = f[k][()]
                else:
                    cls.data[k] = f[k][()]
    
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

    def timeOverride( cls , time_steps ):
        """
        This method overrides the time steps in the rake object.

        Args:
            time_steps (float): The time steps that will be used. Must be the same size as the
                                    original time steps.
        """

        if len( np.shape( time_steps ) ) > 1:
            raise ValueError("Time steps must be a single dimensional array.")
        
        if len( time_steps ) != len( cls.time_steps ):
            raise ValueError("Time steps must be the same size as the original time steps.")

        cls.time_steps = time_steps

    def closeout( cls ):

        del cls.resampled_output

    def dataToPandas( cls , coords = ['x', 'y', 'z'] ):
        """
        Put the data from the Paraview native format to Pandas. Pandas will be more convenient for
            exporting.

        Note that pandas inherently use single dimensional data, and thus may be best used for
            steady data, and data that is unstructured or 1D.

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

class sweep(dataReader):
    """
        This object is similar to the rake object, except the points will convect along a central
    point. This will use any time dependent-enabled functionality, and thus brings some chaneges to
    the dataReader object usage.

    """

    def __init__(self, anchor, point_distribution, datafile, file_format="vtk", t_lims=None ):
        """
        Initialize the rake object according to the inputs to the file.

        Args:

            anchor (float): This is the point where the sweep will convect along during the time
                                span.

            point_distribution ((arrays/lists)):    The points around "anchor" that are in the 
                                                        sweep. Must be in the format:

                                                        [$\Delta x$, $\Delta y$, $\Delta z$]

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

        # Insert inherited class data
        super().__init__( [[0]*2]*3 , datafile , file_format, t_lims=t_lims )

        #
        # Define the points by anchor and point_distribution
        #
        self.points = point_sweep( anchor, point_distribution )
        self.points = np.swapaxes( self.points, 1, 2 )
        anchors = anchor.T[:, None, :] 
        self.frozen_points = self.points - anchors
        
        # Set the object to use time-dependent data
        self.time_dependent=True



class rake(dataReader):
    """
    This object is a rake of points that allows the user to draw data from the datafiles to draw
        the wanted data.
    
    """
    def __init__( self , points , datafile , file_format="vtk" ):
        """
        Initialize the rake object according to the inputs to the file.

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

        # Insert inherited class data
        super().__init__( points , datafile , file_format )

                     

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

class structuredGrid(dataReader):
    """
        In this object, the data will be read via a structured grid. 

        The advantage of this object is that spatial gradients are much easier to calculate this 
    way.

    """

    def __init__(self, points, datafile, file_format="vtk" ):
        """
        Initialize the structured grid object according to the inputs to the file.

        Args:
            points (float): This is a list/array of the points in the structured grid. The order
                                will be (X, Y, Z). All three dimension are required. X, Y, and Z 
                                are the points in a NumPy meshgrid() output-like format.

            datafile (string):  The datafile with the CFD data.

            file_format (string, optional): The file format that will be used. The valid options 
                                                are:

                                            - *"vtk" - The default *.vtk output as OpenFOAM 
                                                        produces
                                            
                                            - "h5" - The *.h5 output that is defined by the 

                                            - None - Take the file format from the "datafile"
                                                        argument.
        """

        # Store the received grid and convert for the dataReader
        self.points_input = points
        self.points_shape = np.shape( self.points_input )[1:]
        X_flat = self.points_input[0].flatten()
        Y_flat = self.points_input[1].flatten()
        Z_flat = self.points_input[2].flatten()

        # Initialize the dataReader object to inherit
        super().__init__( [ X_flat, Y_flat, Z_flat ], datafile, file_format )

        # Move point data
        self.points = [ X_flat, Y_flat, Z_flat ]

    def reform(cls ):
        """
            This method puts the data back in the shape of the input points to make grid 
        calculations easier.

        """

        reform_shape = ( ( len( cls.time_steps ) ,) + cls.points_shape )

        for k in list( cls.data.keys() ):
            print(f"Looking at key {k}")
            cls.data[k] = np.reshape( cls.data[k], reform_shape )


    def gradients(cls ):
        """
            This method calculates the gradients in all directions according to the points the
        object has

        """

        if cls.points_shape[-1]==1:
            grad_raw = np.gradient( cls.points_input, axis=(0,1) )
            cls.points_gradients = np.asarray( [grad_raw[i][i] for i in range(2)] )
        else:
            grad_raw = np.gradient( cls.points_input )
            cls.points_gradients = np.asarray( [grad_raw[i][i] for i in range(3)] )

        # TODO: Fix this so that it take less memory
    

    

    
