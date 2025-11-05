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
    print(f"Anchor shape: {a_shape}, Point Delta shape: {d_shape}")
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
        self.read_raw=False

        # Set the time limits
        self.t_lims = t_lims

    def foamCaseRead( cls, working_dir, file_name="foam.foam", verbosity=0, vector_headers=["U"], 
                     coordinate_system=['x', 'y', 'z'], interpolator="rbf", accelerator=None, headers_read=None, N_sourcePts=1000, allow_dim_drop=True ):
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
        if verbosity>0:
            print(f"Number of dimensions: {N_dims}")
            print(f"Coordinate system: {coordinate_system}")

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
        cls.time_steps = np.array( cls.foam.TimestepValues )

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

        # Limit the time steps
        if cls.t_lims:
            #print(f"Time limits:\t[{np.min(cls.t_lims)}, {np.max(cls.t_lims)}]")
            t_steps = cls.time_steps[ (cls.time_steps >= np.min(cls.t_lims)) & (cls.time_steps <= np.max(cls.t_lims)) ]

        else:
            t_steps = np.array( cls.time_steps )
        #print(f"Original time steps:\t{cls.time_steps}")
        #print(f"Filtered time steps:\t{t_steps}")

        for j, t in enumerate( t_steps ):
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
                if 0.0 in np.sum( np.abs(cell_coords[:,:N_dims]) , axis=0 ) and allow_dim_drop:
                    #print("Actually, it's 1D")
                    drop_dim = np.argmin( np.abs( np.sum( cell_coords[:,:N_dims] , axis=0 ) ) )
                    print(f"Dropping dimension {drop_dim}")
                    object_data = sint.RBFInterpolator( np.delete( cell_coords[:,:N_dims], drop_dim, axis=1 ), data_matrix.T, neighbors=N_sourcePts )( np.delete( cls.points[:,:N_dims], drop_dim, axis=1 ) )
                else:
                    object_data = sint.RBFInterpolator( cell_coords[:,:N_dims], data_matrix.T, neighbors=N_sourcePts )( cls.points[:,:N_dims] )
            elif interpolator.lower() in ["l", "lin", "linear", "linearnd", "delaunay", "delaunaytriangulation"]:
                if 0.0 in np.sum( np.abs(cell_coords[:,:N_dims]) , axis=0 ):
                    #print("Actually, it's 1D")
                    drop_dim = np.argmin( np.abs( np.sum( cell_coords[:,:N_dims] , axis=0 ) ) )
                    print(f"Dropping dimension {drop_dim}")
                    #x = np.delete( cell_coords[:,:N_dims], drop_dim, axis=1 ).reshape( np.shape(np.delete( cell_coords[:,:N_dims], drop_dim, axis=1 ))[0] )
                    #y = data_matrix.T
                    #x_new = np.delete( cls.points[:,:N_dims], drop_dim, axis=1 ).reshape( np.shape(np.delete( cls.points[:,:N_dims], drop_dim, axis=1 ))[0] )
                    #print(f"x is shape {np.shape(x)} and y is shape {np.shape(y)}")
                    #print(f"x_new is shape {np.shape(x_new)} in [{np.min(x_new)}, {np.max(x_new)}]")
                    object_data = sint.LinearNDInterpolator( np.delete( cell_coords[:,:N_dims], drop_dim, axis=1 ), data_matrix.T )( np.delete( cls.points[:,:N_dims], drop_dim, axis=1 ) )
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
        
    def convergeH5DataRead( cls , working_dir , data_prefix="data_ts" , sig_figs=6 , N_dims=3 , interpolator="RBF" , overwrite=False , write=True , rm_after_read=False , mp_method=None , N_cores=None, accelerator=None, headers_exclude=[], fill_value=None ):
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
        # Get original directory
        og_dir = os.getcwd()
        
        # Import needed modules
        import paraview.simple as pasi
        import pandas as pd
        if not accelerator:
            import scipy.interpolate as sint
        elif accelerator.lower() in ["cuda", "cu", "c", "cupy"]:
            import cupyx.scipy.interpolate as sint
        else:
            raise ValueError( "Invalid accelerator engine selected" )
        
        # Move to the working directory
        os.chdir( working_dir )
        
        # Load the data from the files
        cls.data = pasi.CONVERGECFDReader(FileName=cls.file_list[0])
        cls.data.SMProxy.SetAnnotation("ParaView::Name", "MyData")
        print("Available data attributes:\t"+str(dir(cls.data)))

        # Get available time steps
        cls.time_steps = cls.data.TimestepValues
        print(f"Time steps available: {cls.time_steps}")

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
                    """
                    if h not in headers_exclude:
                        col_data += [cls.df_data[h].to_numpy()]
                    """
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

    def hdf5DataRead(cls, working_dir, group_path=["STREAM_00","CELL_CENTER_DATA"], coord_prefix="XCEN", dims=['x','y','z'], interpolator="lin", coords_system=['x','y','z'], mp_method=None, headers_exclude=[], verbosity=1 ):
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
        cls.N_dims = N_dims
        cls.dims = dims

        #
        # Switch between interpolation reading and raw reading
        #
        if not cls.read_raw:

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
            
            # Find the number of non-excluded headers
            to_keep = np.setdiff1d(data_keys, headers_exclude)
            keep_count = to_keep.size
            cls.to_keep = to_keep

            #
            # Go through the time steps and interpolate the data
            #
            if not mp_method:
                for i in range( len(cls.time_steps) ):
                    
                    fl_nm = cls.file_list[i]
                    t_step = cls.time_steps[i]
                    if verbosity>0:
                        print(f"Working on data index:\t{i}")
                        print(f"Working with file:\t{fl_nm}")


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
                        if verbosity>1:
                            print(f"Keys available:\t{keys}")

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
                        for j in range( len( to_keep ) ):
                            if True:
                                data_raw[:,j] = group[to_keep[j]][:]
                        #mask = ~(np.all(data_raw == 0, axis=1) )#| np.isnan(data_raw).all(axis=1))
                        #data_raw = data_raw[mask]
                            
                        
                    if verbosity>1:
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
                    if verbosity>1:
                        print("**Interpolation complete**")

                    #
                    # Move data into dictionary
                    #
                    for j in range( len( to_keep ) ):
                        k = to_keep[j]
                        #print(f"Depositing data for keys {k}:")
                        #print(f"\t{data_array[i,:,j]}")
                        cls.data[k][i,:] = data_array[i,:,j]
                        #print(f"\t{cls.data[k]}")
                        #print(f"\tMin Data:\t{np.min(cls.data[k][i,...])}")
                        #print(f"\tMax Data:\t{np.max(cls.data[k][i,...])}")
                        #print(f"\tMean Data:\t{np.mean(cls.data[k][i,...])}")

            else:
                raise ValueError("Invalid multiprocessing method chosen.")
            
        else:

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
            data_file_path = "/".join(group_path)
            if not cls.time_dependent:
                with h5.File( cls.file_list[0], 'r') as f:

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
                    cls.points = coordinates
                data_shape = ( len(cls.time_steps) ,) + ( len( coordinates[:,0] ) ,)
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
            for i in range( len(cls.time_steps) ):
                print(f"Working on data index:\t{i}")
                fl_nm = cls.file_list[i]
                print(f"Working with file:\t{fl_nm}")
                t_step = cls.time_steps[i]

                #
                # Open the h5 file and deposit data
                #
                with h5.File( fl_nm, 'r') as f:

                    # Pull the data into the group
                    data_file_path = "/".join(group_path)
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
                # Move data into dictionary
                #
                for j in range( len( data_keys ) ):
                    k = data_keys[j]
                    #print(f"Depositing data for keys {k}:")
                    #print(f"\t{data_array[i,:,j]}")
                    cls.data[k][i,:] = data_raw[:,j]
                    #print(f"\t{cls.data[k]}")
                    #print(f"\tMin Data:\t{np.min(cls.data[k][i,...])}")
                    #print(f"\tMax Data:\t{np.max(cls.data[k][i,...])}")
                    #print(f"\tMean Data:\t{np.mean(cls.data[k][i,...])}")

    def hdf5Write( cls , filename , working_dir, skip_headers=[] ):
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

    def __init__(self, anchor, point_distribution, datafile, file_format="vtk", t_lims=None, store_anchors=True, offset=0 ):
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
        self.points = point_sweep( anchor, point_distribution ) + offset
        self.points = np.swapaxes( self.points, 1, 2 )
        anchors = anchor.T[:, None, :] 
        self.deltas = point_distribution
        if store_anchors:
            self.anchors = anchors
        self.frozen_points = self.points - anchors - offset
        
        # Set the object to use time-dependent data
        self.time_dependent=True

    def anchorCorrection(cls, data_dir, scanWindowWidth, N, rel_tol=1e-18, abs_tol=1e-9, read_method="hdf5dataread", 
                         centering_method="edge", centering_header="Mach", input_args=(), iterations=1, iter_mult=2, 
                         edge_dict={"method": "dwt", "wt_family": "bior1.3", "level": -1, "coeff_index": 0, "store_wavelet": True, "dims": ['t', 'y'] } ):
        """
            This method corrects the anchor points of the sweep by defined method.

        Args:
            data_dir (str): This is the directory where the data is located.

            scanWindowWidth (float, width): The limits of the scan window that is used to find the 
                                        correction. Note that these are inclusive limits. They are
                                        in the format:

                                        [min, max] for [x, y, z] coordinates.

            rel_tol (float, optional):  The relative tolerance, or minimum change for the 
                                        correction to continue for the centering. Defaults to 
                                        1e-18.

            abs_tol (float, optional):  The absolute tolerance, or maximum acceptable error for the
                                        centering. Defaults to 1e-9.    

            read_method (str, optional):    The method that the data is read by. The valid options
                                            are:
                                            
                                            - "hdf5dataread" - The default method that reads
                                                the data from the *.h5 files.
                                            
                                            These options are the names of the methods used to read
                                            the data. Not case sensitive. Defaults to 
                                            "hdf5dataread".

            centering_method (str, optional):    The method that the centering is done by. The valid
                                                    options are:

                                                - *"edge" - The default method that uses an edge 
                                                            finding method.
                                                        
                                                - "center" - The method that uses the center of the
                                                            data to center the sweep.

                                                Not case sensitive. Defaults to "edge".

            centering_header (str, optional):    The header that is used to center the sweep. This
                                                    is the header that is used to find the edge or
                                                    center of the data. Defaults to "Mach".

            input_args (tuple, optional):   The input arguments that are used to read the data on 
                                            top of the data_dir.

        """
        print("Under construction.")

        #
        # Set up the correction data in a dictionary
        #
        cls.correction_dict = {}
        cls.correction_dict["abs_tol"] = abs_tol
        cls.correction_dict["rel_tol"] = rel_tol
        cls.correction_dict["read_method"] = read_method.lower()
        cls.correction_dict["centering_method"] = centering_method.lower()
        cls.correction_dict["centering_header"] = centering_header
        cls.correction_dict["scanWindowWidth"] = np.array( scanWindowWidth )
        cls.correction_dict["N"] = N
        cls.correction_dict["Iterations"] = iterations
        cls.correction_dict["Iteration Multiplier"] = iter_mult

        # Store the data directory
        cls.data_dir = data_dir
        cls.input_args = input_args
        cls.edge_dict = edge_dict

        # Back up the old points
        cls.old_points = cls.points.copy()
        cls.errors = []

        #
        # Correct the anchor points
        #
        if cls.correction_dict["centering_method"] in ["edge", 'e']:
            for i in np.arange( cls.correction_dict["Iterations"] ):
                if i>0:
                    cls.correction_dict["scanWindowWidth"] = cls.correction_dict["scanWindowWidth"]/cls.correction_dict["Iteration Multiplier"]
                cls.anchorCorr_xEdgeFind()
            cls.edge_dict = edge_dict
        else:
            raise ValueError("Invalid centering method selected. Only 'edge' is currently supported.")
        
        #
        # Re-do the sweep based on the correct anchor points
        #
        cls.points = point_sweep( cls.anchors[:,0,:].T, cls.deltas )
        cls.points = np.swapaxes( cls.points, 1, 2 )
        cls.newSweep()


    def newSweep(cls ):
        """
            Calculate the new sweep.

        """


        #
        # Create a new sweep along the span
        #
        if cls.correction_dict["read_method"]=="hdf5dataread":
            input_args = ( cls.data_dir, ) + cls.input_args 
            cls.hdf5DataRead( *input_args , dims=cls.dims )
        else:
            raise ValueError("Invalid read method selected. Only 'hdf5dataread' is currently supported.")

    def anchorCorr_xEdgeFind(cls ):
        """
            This method corrects the anchor points of the sweep by edge detection. This is not
        meant to be a standalone method, but rather a method that is used within the object by
        anchorCorrection().

        """
        print("Under construction.")

        # Calculate some important factors
        w = cls.correction_dict["scanWindowWidth"]
        cls.correction_dict["windowArcLength"] = np.linalg.norm( [ np.max(w[0]) - np.min(w[0]) , np.max(w[1]) - np.min(w[1]) , np.max(w[2]) - np.min(w[2]) ] )

        #
        # Define the scan area
        #
        belt = []
        for i in range( len( cls.correction_dict["scanWindowWidth"] ) ):
            belt += [ np.linspace( np.min( w[i] ) , np.max( w[i] ) , num=cls.correction_dict["N"] ) ]

        belt = np.array( belt )
        cls.correction_dict["belt"] = belt
        # Find the span of points based on the minimum distance
        span = point_sweep( np.squeeze( cls.anchors, axis=1 ).T , belt )
        span = np.swapaxes( span, 1, 2 )
        # Find the span of the points
        cls.points = span

        #
        # Find the new sweep
        #
        cls.newSweep()

        #
        # Store the old data
        #
        cls.old_anchors = cls.anchors

        #
        # Find the edge in the data
        #
        if cls.edge_dict["method"] in ["wavelet", "dwt"]:
            print("Using wavelet method to find the edge.")


            # Initialize the compressible gas object and track the shock
            from fluids import compressibleGas
            cls.cmp = compressibleGas( dims=cls.edge_dict["dims"] )
            new_shock_loc, _ = cls.cmp.shockTracking( cls.data, cls.points.T, cls.time_steps, key=cls.correction_dict["centering_header"], store_wavelet=True )
            

            # Now find where the shock is
            if 'x' in cls.edge_dict["dims"]:
                coords = cls.anchors[:,:,0].T
            if 'y' in cls.edge_dict["dims"]:
                coords = cls.anchors[:,:,1].T
            shock_loc_error = np.linalg.norm( coords - new_shock_loc )
            cls.old_shock_loc = cls.anchors
            shock_loc = new_shock_loc
            print(f"New shock location shape:\t{new_shock_loc[0].shape}")
            if 'x' in cls.edge_dict["dims"]:
                cls.anchors[:,:,0][:,0] = new_shock_loc[0]
            if 'y' in cls.edge_dict["dims"]:
                cls.anchors[:,:,1][:,0] = new_shock_loc[0]

            """
            import pywt

            shock_loc = np.zeros( ( len( cls.time_steps ) , 3 ) )
            shock_loc_error = np.zeros( len( cls.time_steps ) )
            for i in range( len( cls.time_steps ) ):

                #
                # Calculate the wavelet transform
                #
                if cls.edge_dict["level"]==-1:
                    decomp = pywt.wavedec( cls.data[cls.correction_dict["centering_header"]][i] , wavelet=cls.edge_dict["wt_family"] )
                    
                else:
                    decomp = pywt.wavedec( cls.data[cls.correction_dict["centering_header"]][i] , wavelet=cls.edge_dict["wt_family"], level=cls.edge_dict["level"] )
                if cls.edge_dict["store_wavelet"]:
                    cls.correction_dict["wavelet_coeffs"] = decomp
                
                #
                # Find the next best guess of a location of the shock
                #
                try:
                    shock_loc_indx = np.nanargmax( decomp[cls.edge_dict["level"]][1:-1] )
                    shock_loc[i] = cls.points[i, 2*(shock_loc_indx+1), :]
                except:
                    print(f"Shock location finding failed, { decomp[cls.edge_dict['level']][1:-1] }")
                    shock_loc_indx = 0
                    shock_loc[i] = cls.anchors[i]
                shock_loc_error[i] = np.linalg.norm( cls.old_anchors[i] - shock_loc[i] )
            #"""
                
            cls.shock_loc = shock_loc
            cls.rel_error = shock_loc_error
            cls.errors += [cls.rel_error]
            



        else:
            raise ValueError("Invalid edge detection method selected. Only 'wavelet' is currently supported in the edge dictionary.")
        


    def frozenData(cls, lims=None ):
        """
            This method takes the data present and freezes it in time. The coordinate system for 
        the data now is .frozen_points.

        """

        import scipy.stats as scst

        cls.frozen_data = {}
        cls.variance_data = {}
        cls.skewness_data = {}
        cls.kurtosis_data = {}
        for i, k in enumerate( list( cls.data.keys() ) ):
            if not lims:
                cls.frozen_data[k] = np.mean( cls.data[k], axis=0 )
                cls.variance_data[k] = np.var( cls.data[k], axis=0 )
                cls.skewness_data[k] = scst.skew( cls.data[k], axis=0 )
                cls.kurtosis_data[k] = scst.kurtosis( cls.data[k], axis=0 )
            else:
                cls.frozen_data[k] = np.mean( cls.data[k][np.min(lims):np.max(lims)], axis=0 )
                cls.variance_data[k] = np.var( cls.data[k][np.min(lims):np.max(lims)], axis=0 )
                cls.skewness_data[k] = scst.skew( cls.data[k][np.min(lims):np.max(lims)], axis=0 )
                cls.kurtosis_data[k] = scst.kurtosis( cls.data[k][np.min(lims):np.max(lims)], axis=0 )

class rake(dataReader):
    """
    This object is a rake of points that allows the user to draw data from the datafiles to draw
        the wanted data.
    
    """
    def __init__( self , points , datafile , file_format="vtk", read_raw=False ):
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

            read_raw (boolean, optional):  Whether to read the raw data from the file. Recommended
                                                only to use for 1D cases. Defaults to False.

        Attributes:

            ext_points [list]:  The externally defined points from "points" re-formatted into a
                                    Paraview-friendly format.
        
        """

        # Insert inherited class data
        super().__init__( points , datafile , file_format )

        # Set the read raw values
        self.read_raw = read_raw

                     

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
        self.points = np.array([ X_flat, Y_flat, Z_flat ])

    def reform(cls, verbosity=0 ):
        """
            This method puts the data back in the shape of the input points to make grid 
        calculations easier.

        """

        reform_shape = ( ( len( cls.time_steps ) ,) + cls.points_shape )

        for k in list( cls.data.keys() ):
            if verbosity>0:
                print(f"Re-forming key {k}")
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

    def timeIntegration(cls, ti_lims=None ):

        cls.data_timeIntegrated = {}
        for i, k in enumerate( list( cls.data.keys() ) ):
            if not ti_lims:
                cls.data_timeIntegrated[k] = np.trapz( cls.data[k], cls.time_steps, axis=0 )
            else:
                cls.data_timeIntegrated[k] = np.trapz( cls.data[k][np.min(ti_lims):np.max(ti_lims)], cls.time_steps[np.min(ti_lims):np.max(ti_lims)], axis=0 )
            

    def writeImage(cls, working_dir, write_timeIntegration=True, write_keys=None ):

        import matplotlib.pyplot as plt

        os.chdir( working_dir )

        if write_timeIntegration:
            for i, k in enumerate( list( cls.data.keys() ) ):
                if not write_keys:
                    plt.imsave( f"output_timeIntegration_{k}.png", cls.data_timeIntegrated[k], cmap="gray" )
                elif k in write_keys:
                    plt.imsave( f"output_timeIntegration_{k}.png", cls.data_timeIntegrated[k], cmap="gray" )


    def cellMeasurement(cls, working_dir, image_prefix, data_keys=None, debug=1 ):

        from external.soot_foil_image_tool import measure_image
        import glob, os
        from natsort import natsorted
        
        # Move to the directory and pull all the images
        os.chdir( working_dir )
        img_paths = natsorted( glob.glob( f"{image_prefix}*" ) )
        print(f"Seeing {img_paths} available")

        # Calculate the bounds that the image represents
        Dx = np.max( cls.points_input[0] ) - np.min( cls.points_input[0] )
        Dy = np.max( cls.points_input[1] ) - np.min( cls.points_input[1] )

        # Read the image
        cls.annotateds = []
        cls.measurements = []
        for i, img in enumerate( img_paths ):
            annotated, measurement = measure_image( img, 'w', Dy*100, debug=debug, verbosity=2 )
            cls.annotateds += [annotated]
            cls.measurements += [measurement/100]


class particles_EulerLagrange(dataReader):
    """
        This object is meant to handle the behavior of particles that are present in a simulation
    using an Euler-Lagrange approach.

    """
    def __init__(self, datafile, file_format="vtk", data_directory=None ):
        """
        Initialize the particles object according to the inputs to the file.

        Note that we do not have a set of points yet, as these will come from the particle
        locations. 

        Args:
            datafile (string):  The datafile with the CFD data.

            file_format (string, optional): The file format that will be used. The valid options 
                                                are:

                                            - *"vtk" - The default *.vtk output as OpenFOAM 
                                                        produces
                                            
                                            - "h5" - The *.h5 output that is defined by the 

                                            - None - Take the file format from the "datafile"
                                                        argument.
        """
        # If a data directory is given, move to it
        if data_directory:
            og_dir = os.getcwd()
            os.chdir( data_directory )
            self.data_dir = data_directory   

        # Initialize the dataReader object to inherit
        super().__init__( [[],[],[]], datafile, file_format )

        # Move back to the original directory
        if data_directory:
            os.chdir( og_dir )

    def convergeParticleReader(cls, working_dir, accelerator=None, group_path=["STREAM_00","PARCEL_DATA","LIQUID_PARCEL_DATA"], particle_prefix="LIQPARCEL" ):        
        """
            This method reads the particle data from a Converge simulation output file set. The
        default arguments are to be as close to Converge's default output as possible.

        Args:
            working_dir (string):   The directory to work the data from.

            accelerator (string, optional):    The accelerator that will be used to speed up
                                                    interpolation. The valid options are:

                                                - *None:   The default option that uses SciPy
                                                            interpolation.

                                                - "cuda" or "cupy" or "cu" or "c":   The
                                                            option that uses CuPy to accelerate the
                                                            interpolation. Note that this requires
                                                            a CUDA-capable GPU and the proper
                                                            installation of CuPy with CUDA
                                                            support.

        """

        print("Under construction.")

        # Get the original directory
        og_dir = os.getcwd()

        # Import the needed libraries
        import paraview.simple as pasi
        import pandas as pd
        import h5py as h5
        if not accelerator:
            import scipy.interpolate as sint
        elif accelerator.lower() in ["cuda", "cu", "c", "cupy"]:
            import cupyx.scipy.interpolate as sint
        else:
            raise ValueError( "Invalid accelerator engine selected" )
        
        # Move to the working directory
        os.chdir( working_dir )
        
        # Load the data from the files
        cls.data_sv = pasi.CONVERGECFDReader(FileName=cls.file_list[0])
        cls.data_sv.SMProxy.SetAnnotation("ParaView::Name", "MyData")
        print("Available data attributes:\t"+str(dir(cls.data_sv)))

        # Get available time steps
        cls.time_steps = cls.data_sv.TimestepValues
        print(f"Time steps available: {cls.time_steps}")

        # Iterate over the time steps and pull the data from the h5 files
        for i, t in enumerate( cls.time_steps ):
            print(f"Reading time step {t:.3e} at index {i}")

            # Set the data path
            dataPath = ""
            for gp in group_path:
                dataPath += f"{gp}/"
            dataPath = dataPath[:-1]
            print(f"Data path:\t{dataPath}")

            # Initialize the data dictionary, if first time step
            if i==0:
                cls.data = {}

                # We will initialize the data keys from the first file and the first parcel 
                #   available
                with h5.File( cls.file_list[i], 'r') as f:
                    dset = f[dataPath]
                    pkeys = list(dset.keys())
                    #print(f"Parcel keys available:\t{pkeys}")

                    dkeys = list(dset[pkeys[0]].keys())
                    #print(f"Data keys available:\t{dkeys}")
                    for dk in dkeys:
                        cls.data[dk] = [[]]
            else:
                for dk in list(cls.data.keys()):
                    cls.data[dk] += [[]]

            

            # Pull data into the data dictionary
            """
                Note the heirarchy of the data is as follows:
            [data key][time step][parcel ID/number]

            """
            with h5.File( cls.file_list[i], 'r') as f:
                dset = f[dataPath]
                pkeys = list(dset.keys())
                #print(f"Parcel keys available:\t{pkeys}")
                
                for pk in pkeys:
                    data_set = dset[pk]
                    #print(f"Data set keys available for parcel {pk}:\t{list(data_set.keys())}")
                    for dk in list( cls.data.keys() ):
                        raw_data = data_set[dk][:]
                        #print(f"\tData key {dk} has shape {np.shape(raw_data)}")
                        #print(f"\tData key {dk} has data {raw_data}")
                        if np.shape(raw_data)[0]==1:
                            cls.data[dk][i] += raw_data.tolist()
                        else:
                            cls.data[dk][i] += raw_data.tolist()

        # Move back to the original directory
        os.chdir( og_dir )


    def convergeSurroundingsReader(cls, group_path=["STREAM_00","CELL_CENTER_DATA"], coord_prefix="PARCEL_", source_coord_prefix="XCEN_", dims=['x','y','z'], interpolator="lin", coords_system=['x','y','z'], mp_method=None, headers_exclude=[], verbosity=1, neighbors=1000, store_source_data=False ):
        """
            This method finds the data for the surroundings of the particles in a Converge 
        simulation. This is largely a clone of hdf5DataRead from dataReader, but modified for the
        different data structure.

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
        og_dir = os.getcwd()
        os.chdir( cls.data_dir )

        import h5py as h5
        import scipy.interpolate as sint

        # Set number of dimension
        N_dims = len( dims )
        cls.N_dims = N_dims
        cls.dims = dims


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
        # Set up the coordinates into the list that corresponds to the data structure
        #
        parcel_coords = []
        cls.coords_list = [
                            cls.data[coord_key] 
                            for coord_key in cls.data.keys() 
                            if coord_prefix in coord_key
                        ]


        # Initialize our data
        data_file_path = "/".join(group_path)
        with h5.File( cls.file_list[1], 'r') as f:
            group = f[data_file_path]
            keys = list(group.keys())
            data_keys = [key for key in keys if not key.startswith(coord_prefix)]
        print(f"Working with data keys:\t{data_keys}")
        
        # Find the number of non-excluded headers
        to_keep = np.setdiff1d(data_keys, headers_exclude)
        keep_count = to_keep.size
        cls.to_keep = to_keep

        # Initialize data dictionary for surrounding data
        cls.data_surrounding = {}
        for k in data_keys:
            cls.data_surrounding[k] = []
            for i in range( len( cls.time_steps) ):
                cls.data_surrounding[k] += [[]]
                for j in range( len( cls.data[ list(cls.data.keys())[0] ][i] ) ):
                    cls.data_surrounding[k][i] += [0]

        #
        #   Iterate through the time steps and pull the data
        #
        cls.og_data = {}
        cls.data_sources = []
        for i in range( len( cls.time_steps) ):
            print(f"Reading time step {cls.time_steps[i]:.3e} at index {i}")

            # Pull data into source data matrix
            #source_data = np.zeros( ( len( list(cls.data.keys()) ) ,) + )
            
            #
            # Get the coordinates of the parcels at this time step
            #
            N_parcels = len( cls.data[ list(cls.data.keys())[0] ][i] )
            target_coords = np.zeros( ( N_parcels, 3) )
            for j in range( N_parcels ):
                print(f"\tReading parcel {j+1} of {N_parcels}")
                
                target_coords[j,0] = cls.data[coord_prefix+"X"][i][j]
                target_coords[j,1] = cls.data[coord_prefix+"Y"][i][j]
                target_coords[j,2] = cls.data[coord_prefix+"Z"][i][j]

            if N_parcels>0:
                #
                # Get the source coordinates and data
                #
                with h5.File( cls.file_list[i], 'r') as f:
                    group = f[data_file_path]
                    keys = list(group.keys())
                    #print(f"Available groups:\t{keys}")
                    N_source_points =  len( group[ source_coord_prefix+"X" ][:] )
                    source_coords = np.zeros( ( len( cls.coords_list ) , N_source_points ) ).T
                    source_coords[:,0] = group[ source_coord_prefix+"X" ][:]
                    source_coords[:,1] = group[ source_coord_prefix+"Y" ][:]
                    source_coords[:,2] = group[ source_coord_prefix+"Z" ][:]                
                    
                    # Pull the data to interpolate
                    source_data = np.zeros( ( len( data_keys ) , N_source_points ) ).T
                    for k, dk in enumerate( data_keys ):
                        source_data[:,k] = group[ dk ][:]
                        if store_source_data:
                            if not dk in cls.og_data.keys():
                                cls.og_data[dk] = []
                            cls.og_data[dk] += [ group[ dk ][:] ]

                    if store_source_data:    
                        cls.data_sources += [ source_data[:,k] ]
                
                #print(f"\t\tSource coords:\t{source_coords}")
                #print(f"\t\tTarget coords:\t{target_coords}")
                #print(f"\t\tSource data:\t{source_data}")
                #
                # Interpolate the data to the parcel locations
                #
                if interpolator.lower() in ["rbf", "r"]:
                    interpolator_object = sint.RBFInterpolator( source_coords, source_data, neighbors=int(neighbors) )
                elif interpolator.lower() in ["linear", "lin", "l"]:
                    interpolator_object = sint.LinearNDInterpolator( source_coords, source_data )
                
                raw_interpOut = interpolator_object( target_coords )
                #print(f"\t\tRaw interpolated output:\t{raw_interpOut}")
                for k, dk in enumerate( data_keys ):
                    cls.data_surrounding[dk][i] = raw_interpOut[:,k]

    def velocities(cls, parcel_prefix="VELOCITY_", flow_prefix="VELOCITY_" ):
        """
            This method calculates the velocities that pertain to the particles. The velocites 
        keep the standard of turbomachinery as follows:

            - U: The velocity of the particle in the absolute reference frame.
            - C: The velocity of the flow in the absolute reference frame.
            - W: The velocity of the particle in the relative reference frame. W = C-W.

        """

        cls.data["U"] = []
        cls.data["C"] = []
        cls.data["W"] = []
        for i in range( len( cls.time_steps ) ):
            timeSte_us = []
            timeSte_cs = []
            timeSte_ws = []
            for j in range( np.array(cls.data[list(cls.data.keys())[0]][i]).shape[0] ):
                u = np.array( [ cls.data[parcel_prefix+"X"][i][j], cls.data[parcel_prefix+"Y"][i][j], cls.data[parcel_prefix+"Z"][i][j] ] )
                c = np.array( [ cls.data_surrounding[flow_prefix+"X"][i][j], cls.data_surrounding[flow_prefix+"Y"][i][j], cls.data_surrounding[flow_prefix+"Z"][i][j] ] )
                w = c - u
                timeSte_us += [u]
                timeSte_cs += [c]
                timeSte_ws += [w]
            cls.data["U"] += [timeSte_us]
            cls.data["C"] += [timeSte_cs]
            cls.data["W"] += [timeSte_ws]
                


    def collect_data(cls, surface_tension=None ):
        """
            This method collects various pieces of data that are useful for particle analysis. This includes:

            - Reynolds number
            - Weber number
            - Drag coefficient

        """

        #
        # Get Drag Force
        #
        cls.data["accel"] = []
        cls.data["r"] = []
        cls.data["F_D"] = []
        cls.data["Volume"] = [] 
        N_drops_MAX = 0
        for i in range( len( cls.time_steps ) ):
            timeSte_rs = []
            timeSte_Vs = []
            for j in range( np.array(cls.data[list(cls.data.keys())[0]][i]).shape[0] ):
                r = np.array( [ cls.data["PARCEL_X"][i][j], cls.data["PARCEL_Y"][i][j], cls.data["PARCEL_Z"][i][j] ])
                timeSte_rs += [r]
                timeSte_Vs += [ 4 * np.pi * cls.data["RADIUS"][i][j]**3 / 3 ]
            cls.data["r"] += [timeSte_rs]
            cls.data["Volume"] += [timeSte_Vs]
            if N_drops_MAX < np.array(cls.data[list(cls.data.keys())[0]][i]).shape[0]:
                N_drops_MAX = np.array(cls.data[list(cls.data.keys())[0]][i]).shape[0]
        for j in range( N_drops_MAX ):
            rs_ = []
            for i in range( len( cls.time_steps ) ):
                if j < np.array(cls.data[list(cls.data.keys())[0]][i]).shape[0]:
                    rs_ += [ cls.data["r"][i][j] ]
            rs_ = np.array( rs_ )
            cls.data["accel"] += [np.gradient( np.gradient( rs_, cls.time_steps[:len(rs_)], axis=0 ), cls.time_steps[:len(rs_)], axis=0 )]
        for i in range( len( cls.time_steps ) ):
            timeSte_FDs = []
            for j in range( np.array(cls.data[list(cls.data.keys())[0]][i]).shape[0] ):
                timeSte_FDs += [ cls.data["MASS"][i][j] * cls.data["accel"][j][i] ]
            cls.data["F_D"] += [timeSte_FDs]

        #
        # Get target data
        #
        cls.data["Re"] = []
        cls.data["We"] = []
        cls.data["We_droplet"] = []
        cls.data["C_D"] = []
        for i in range( len( cls.time_steps ) ):
            timeSte_Res = []
            timeSte_Wes = []
            timeSte_Weds = []
            timeSte_CDs = []
            for j in range( np.array(cls.data[list(cls.data.keys())[0]][i]).shape[0] ):
                # Get the Reynolds number
                Re = cls.data_surrounding["DENSITY"][i][j] * np.linalg.norm( cls.data["W"][i][j] ) * 2*cls.data["RADIUS"][i][j] / cls.data_surrounding["MOL_VISC"][i][j]
                timeSte_Res += [Re]

                # Get the Weber number relative to the surroundings
                if not surface_tension:
                    We = cls.data_surrounding["DENSITY"][i][j] * np.linalg.norm( cls.data["W"][i][j] )**2 * cls.data["RADIUS"][i][j] / cls.data_surrounding["SURF_TENS"][i][j]
                else:
                    We = cls.data_surrounding["DENSITY"][i][j] * np.linalg.norm( cls.data["W"][i][j] )**2 * cls.data["RADIUS"][i][j] / surface_tension
                timeSte_Wes += [We]

                # Get the Weber number for only the droplets
                if not surface_tension:
                    We = ( cls.data["MASS"][i][j] / cls.data["Volume"][i][j] ) * np.linalg.norm( cls.data["W"][i][j] )**2 * cls.data["RADIUS"][i][j] / cls.data_surrounding["SURF_TENS"][i][j]
                else:
                    We = ( cls.data["MASS"][i][j] / cls.data["Volume"][i][j] ) * np.linalg.norm( cls.data["W"][i][j] )**2 * cls.data["RADIUS"][i][j] / surface_tension
                timeSte_Weds += [We]

                # Get the drag coefficient
                if np.linalg.norm( cls.data["W"][i][j] )>0:
                    C_D = 2 * np.linalg.norm( cls.data["F_D"][i][j] ) / ( cls.data_surrounding["DENSITY"][i][j] * np.linalg.norm( cls.data["W"][i][j] )**2 * np.pi * (cls.data["RADIUS"][i][j])**2 )
                else:
                    C_D = 0
                timeSte_CDs += [C_D]

            cls.data["Re"] += [timeSte_Res]
            cls.data["We"] += [timeSte_Wes]
            cls.data["We_droplet"] += [timeSte_Weds]
            cls.data["C_D"] += [timeSte_CDs]

class fullCV(dataReader):
    """
        This object deviates from the standard data reader to read the entire stored control volume.

    """
    def __init__(self, datafile, file_format="vtk" ):
        """
        Initialize the full CV object according to the inputs to the file.

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

        # Initialize the dataReader object to inherit
        super().__init__( [[],[],[]], datafile, file_format )


    def fullhdf5DataRead(cls, working_dir, group_path=["STREAM_00","CELL_CENTER_DATA"], coord_prefix="XCEN", dims=['x','y','z'], interpolator="lin", coords_system=['x','y','z'], mp_method=None, headers_exclude=[] ):
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
        cls.N_dims = N_dims
        cls.dims = dims

        #
        # Switch between interpolation reading and raw reading
        #
        if not cls.read_raw:

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
            # Initialize our data
            #
            data_file_path = "/".join(group_path)
            with h5.File( cls.file_list[1], 'r') as f:
                group = f[data_file_path]
                keys = list(group.keys())
                data_keys = [key for key in keys if not key.startswith(coord_prefix)]
            cls.data={}
            for k in data_keys:
                cls.data[k]=[]
            
            # Find the number of non-excluded headers
            to_keep = np.setdiff1d(data_keys, headers_exclude)
            keep_count = to_keep.size
            cls.to_keep = to_keep

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
                        print(f"Keys available:\t{keys}")

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
                        for j in range( len( to_keep ) ):
                            if True:
                                data_raw[:,j] = group[to_keep[j]][:]
                        #mask = ~(np.all(data_raw == 0, axis=1) )#| np.isnan(data_raw).all(axis=1))
                        #data_raw = data_raw[mask]
                            
                    print("**Data pull complete**")

                    #
                    # Move data into dictionary
                    #
                    for j in range( len( to_keep ) ):
                        k = to_keep[j]
                        #print(f"Depositing data for keys {k}:")
                        #print(f"\t{data_array[i,:,j]}")
                        cls.data[k] += [data_raw[:,j]]
                        #print(f"\t{cls.data[k]}")
                        #print(f"\tMin Data:\t{np.min(cls.data[k][i,...])}")
                        #print(f"\tMax Data:\t{np.max(cls.data[k][i,...])}")
                        #print(f"\tMean Data:\t{np.mean(cls.data[k][i,...])}")

            else:
                raise ValueError("Invalid multiprocessing method chosen.")
            
        else:

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
            data_file_path = "/".join(group_path)
            if not cls.time_dependent:
                with h5.File( cls.file_list[0], 'r') as f:

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
                    cls.points = coordinates
                data_shape = ( len(cls.time_steps) ,) + ( len( coordinates[:,0] ) ,)
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
            for i in range( len(cls.time_steps) ):
                print(f"Working on data index:\t{i}")
                fl_nm = cls.file_list[i]
                print(f"Working with file:\t{fl_nm}")
                t_step = cls.time_steps[i]

                #
                # Open the h5 file and deposit data
                #
                with h5.File( fl_nm, 'r') as f:

                    # Pull the data into the group
                    data_file_path = "/".join(group_path)
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
                # Move data into dictionary
                #
                for j in range( len( data_keys ) ):
                    k = data_keys[j]
                    #print(f"Depositing data for keys {k}:")
                    #print(f"\t{data_array[i,:,j]}")
                    cls.data[k][i,:] = data_raw[:,j]
                    #print(f"\t{cls.data[k]}")
                    #print(f"\tMin Data:\t{np.min(cls.data[k][i,...])}")
                    #print(f"\tMax Data:\t{np.max(cls.data[k][i,...])}")
                    #print(f"\tMean Data:\t{np.mean(cls.data[k][i,...])}")


    def volumeIntegrals(cls ):
        """
            This method calculates CV integrals that are useful to evaluate the CFD validity.

        """

        # Initialize the time integrals
        cls.volume_integrals = {}
        cls.volume_integrals["Mass"] = np.zeros( len( cls.time_steps ) )
        cls.volume_integrals["Internal Energy"] = np.zeros( len( cls.time_steps ) )
        cls.volume_integrals["Enthalpy"] = np.zeros( len( cls.time_steps ) )

        # Calculate time integrals
        for i, t in enumerate( cls.time_steps ):
            cls.volume_integrals["Mass"][i] = np.sum( cls.data["MASS"][i] )
            cls.volume_integrals["Internal Energy"][i] = np.sum( cls.data["MASS"][i] * cls.data["SIE"][i] )
            cls.volume_integrals["Enthalpy"][i] = cls.volume_integrals["Internal Energy"][i] + np.sum( cls.data["PRESSURE"][i] * cls.data["VOLUME"][i] )


        # Calculate rates
        cls.volume_integral_rates = {}
        cls.volume_integral_rates["Mass Generation"] = np.gradient( cls.volume_integrals["Mass"], cls.time_steps )
        cls.volume_integral_rates["Normalized Mass Generation"] = cls.volume_integral_rates["Mass Generation"] / cls.volume_integrals["Mass"]
        cls.volume_integral_rates["Energy Generation"] = np.gradient( cls.volume_integrals["Enthalpy"], cls.time_steps )





    

    

    
