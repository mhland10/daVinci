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
import sys
import pandas as pd
from natsort import natsorted

#==================================================================================================
#
#   Worker Functions for Multiprocessing
#
#==================================================================================================

def write_worker_function( input_args ):
    """
    This function does the writing of the *.csv's to the working dir.

    Args:
        time_indices_chunk (int):   The list of time indices that will be written
                                        relative to Paraview's time indices.
    """

    import paraview.simple as pasi

    time_indices_chunk , working_dir , data_prefix , write_proxy , data_keys , sig_figs = input_args

    for t_i in time_indices_chunk:
        file_write_nm = working_dir + "\\" + data_prefix + str(t_i) + ".csv"
        print(f"Changing directory to {os.getcwd()}\n to write {file_write_nm}")
        pasi.SaveData( file_write_nm , proxy=write_proxy, 
                    WriteTimeSteps=t_i, WriteTimeStepsSeparately=0, Filenamesuffix='_%d', 
                    ChooseArraysToWrite=0, PointDataArrays=[], 
                    CellDataArrays = data_keys + ["CellCenters"] , 
                    FieldDataArrays=[], VertexDataArrays=[], EdgeDataArrays=[], RowDataArrays=[],
                    Precision=sig_figs, UseStringDelimiter=1, UseScientificNotation=1, 
                    FieldAssociation='Cell Data', AddMetaData=0, AddTimeStep=0, AddTime=0)

    return None

def read_worker_function( input_args ):
    """
    This function does the reading of the *.csv's from the working directory to Pandas.


    Args:
        time_indices_chunk (int):   The list of time indices that will be read relative to 
                                        Paraview's time indices.
    """

    time_indices_chunk , working_dir , data_prefix , rm_after_read = input_args

    # Create empty list to add the dataframes to
    df_read_c = []

    for t_i in time_indices_chunk:
        file_write_nm = working_dir + "\\" + data_prefix + str(t_i) + ".csv"
        print(f"Reading {file_write_nm}")
        # Pull data into dataframe
        df_read_c += [pd.read_csv( file_write_nm,
                            sep=',',  # Use '\t' if your file is tab-delimited
                            header=0,  # The first row as column names
                            )]
        if rm_after_read:
            os.remove(file_write_nm)
        
    return df_read_c       

#==================================================================================================
#
#   Paraview Post-Processing Objects
#
#==================================================================================================

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
        
        # Expand wildcard pattern into actual file list with natural sorting
        self.datafile = datafile
        self.file_list = natsorted(glob.glob(datafile))  # Use natsorted instead of sorted
        print("File list:\t"+str(self.file_list))

        # Store the file format
        self.file_format = file_format

        # Store the points on the rake
        self.ext_points = [[points[0][i], points[1][i], points[2][i]] for i in range(len(points[0]))]

        # Set coordinate change variable to track if the coordinate change of the rake has occured
        self.coord_change=False                

    def paraviewDataRead( cls ):
        """
        This method reads the data using the Paraview engine and stores the data in the rake object
            in a dictionary.

        """

        import paraview.simple as pasi
        import vtk

        # If the file format is inferred, find what it is
        if cls.file_format is None:
            ext = os.path.splitext(datafile)[-1].lower()
            if ext == ".vtk":
                cls.file_format = "vtk"
            elif ext == ".h5":
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
        print("Available data attributes:\t"+str(dir(data)))

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
        print("Resample attributes:\t"+str(dir(resample)))
        pasi.UpdatePipeline()
        resampled_output = pasi.servermanager.Fetch(resample)
        point_data = resampled_output.GetPointData()

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
            point_data = resampled_output.GetPointData()

            # Extract data for each variable and store it
            for header in cls.array_headers:
                array = point_data.GetArray(header)
                if array:
                    cls.data_dict[header].append([array.GetValue(i) for i in range(array.GetNumberOfTuples())])

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

    def convergeH5DataRead( cls , working_dir , data_prefix="data_ts" , sig_figs=6 , N_dims=3 , interpolator="RBF" , overwrite=False , rm_after_read=False , mp_method=None , N_cores=None ):
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

            overwrite (boolean, optional):  Whether the reader will overwrite *.csv files if they
                                                are already present.
            
            rm_after_read (boolean, optional):  Whether the *.csv file that corresponds to a time
                                                    step will be deleted after reading the data
                                                    into a Pandas dataframe. Defaults to False.

            mp_method (string, optional):   Which multiprocessing method to parallelize the
                                                operations. 

        """

        import paraview.simple as pasi
        import pandas as pd
        import scipy.interpolate as sint
        
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

        if not mp_method:

            for t_i , t in enumerate( cls.time_steps ):

                # Save the data to a *.csv
                os.chdir( working_dir )
                file_write_nm = working_dir + "\\" + data_prefix + str(t_i) + ".csv"
                if overwrite or not os.path.exists(file_write_nm):
                    print(f"Changing directory to {os.getcwd()}\n to write {file_write_nm}")
                    pasi.SaveData( file_write_nm , proxy=cls.coordinates1, 
                                WriteTimeSteps=t_i, WriteTimeStepsSeparately=0, 
                                Filenamesuffix='_%d', ChooseArraysToWrite=0, 
                                PointDataArrays=[], 
                                CellDataArrays = cls.source.CellData.keys() + ["CellCenters"] , 
                                FieldDataArrays=[], VertexDataArrays=[], EdgeDataArrays=[], 
                                RowDataArrays=[], Precision=sig_figs, UseStringDelimiter=1, 
                                UseScientificNotation=1, FieldAssociation='Cell Data', 
                                AddMetaData=0, AddTimeStep=0, AddTime=0)
                
                # Pull data into dataframe
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
                cls.rake_coordinates = np.zeros( ( np.shape( cls.ext_points )[0] , N_dims ) )
                for i in range( np.shape( cls.ext_points )[0] ):
                    cls.rake_coordinates[i,:] = cls.ext_points[i][:N_dims]

                # Initialize the dictionary
                if t_i==0:
                    for i , h in enumerate( cls.df_data.columns ):
                        cls.data[h] = np.zeros_like( shape_matrix )

                # Break the data into a dictionary
                # TODO: RBFInterpolator() takes up too much RAM. At some point, we should make an optimized
                #           version that can fit into limited RAM.
                for i , h in enumerate( cls.df_data.columns ):
                    col_data = cls.df_data[h].to_numpy()
                    if interpolator.lower()=="rbf":
                        cls.data[h][t_i,:] = sint.RBFInterpolator( cls.coordinates , col_data )( cls.rake_coordinates )
                    elif interpolator.lower()=="ct" and N_dims==2:
                        cls.data[h][t_i,:] = sint.CloughTocher2DInterpolator( cls.coordinates , col_data )( cls.rake_coordinates )
                    else:
                        raise ValueError("Invalid interpolator selected.")
                    
        elif mp_method.lower()=="mp" or mp_method.lower()=="multiprocessing":

            import multiprocessing as mp                        

            print("**Not currently implemented**")

        elif mp_method.lower()=="mpi":

            print("**Under construction**")

            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            
            print(f"Running on {size} ranks...")

            # Distribute time steps to processes
            time_steps_per_rank = len(cls.time_steps) // size
            print(f"There are {time_steps_per_rank} time steps per rank.")
            cls.time_indices = np.arange(len(cls.time_steps))
            cls.time_steps_chunks = np.array_split( cls.time_steps , time_steps_per_rank )
            cls.time_index_chunks = np.array_split( cls.time_indices , time_steps_per_rank )
            print(f"Time Step chunks: {cls.time_index_chunks}")
            print(f"First time step chunk: {cls.time_index_chunks[0]}")

            # Set up dictionary to receive the rank data
            rank_data = {}

            # Iterate over time steps assigned to the current rank
            for t_i in cls.time_index_chunks[rank]:
                t = cls.time_steps[int(t_i)]

                # Save the data to a *.csv for each time step
                os.chdir(working_dir)
                file_write_nm = working_dir + "\\" + data_prefix + str(t_i) + ".csv"
                
                if overwrite or not os.path.exists(file_write_nm):
                    print(f"Rank {rank} changing directory to {os.getcwd()}\n to write {file_write_nm}")
                    pasi.SaveData(
                        file_write_nm, 
                        proxy=cls.coordinates1,
                        WriteTimeSteps=t_i, 
                        WriteTimeStepsSeparately=0,
                        Filenamesuffix='_%d', 
                        ChooseArraysToWrite=0, 
                        PointDataArrays=[],
                        CellDataArrays=cls.source.CellData.keys() + ["CellCenters"], 
                        FieldDataArrays=[], 
                        VertexDataArrays=[], 
                        EdgeDataArrays=[], 
                        RowDataArrays=[], 
                        Precision=sig_figs, 
                        UseStringDelimiter=1, 
                        UseScientificNotation=1, 
                        FieldAssociation='Cell Data', 
                        AddMetaData=0, 
                        AddTimeStep=0, 
                        AddTime=0
                    )

                # Pull data into DataFrame
                cls.df_read = pd.read_csv(file_write_nm, sep=',', header=0)

                if rm_after_read:
                    os.remove(file_write_nm)

                # Separate into coordinates and data
                data_columns = [col for col in cls.df_read.columns if not col.startswith('CellCenters')]
                cls.df_data = cls.df_read[data_columns]
                coord_columns = [col for col in cls.df_read.columns if col.startswith('CellCenters')]
                cls.df_coord = cls.df_read[coord_columns]

                # Set up coordinates and rake coordinates as numpy arrays
                cls.coordinates = np.zeros((len(cls.df_coord[cls.df_coord.keys()[0]].to_numpy()), N_dims))
                for i in range(N_dims):
                    cls.coordinates[:, i] = cls.df_coord[cls.df_coord.keys()[i]].to_numpy()

                cls.rake_coordinates = np.zeros((np.shape(cls.ext_points)[0], N_dims))
                for i in range(np.shape(cls.ext_points)[0]):
                    cls.rake_coordinates[i, :] = cls.ext_points[i][:N_dims]

                # Store data on the current rank
                rank_data[t_i] = {}

                for i, h in enumerate(cls.df_data.columns):
                    col_data = cls.df_data[h].to_numpy()
                    if interpolator.lower() == "rbf":
                        rank_data[t_i][h] = sint.RBFInterpolator(cls.coordinates, col_data)(cls.rake_coordinates)
                    elif interpolator.lower() == "ct" and N_dims == 2:
                        rank_data[t_i][h] = sint.CloughTocher2DInterpolator(cls.coordinates, col_data)(cls.rake_coordinates)
                    else:
                        raise ValueError("Invalid interpolator selected.")

            # Gather the data from all ranks to rank 0
            all_data = comm.gather(rank_data, root=0)

            # Rank 0 combines the results
            if rank == 0:
                # Initialize the final data dictionary
                cls.data = {}
                
                for rank_data_chunk in all_data:
                    for t_i, data in rank_data_chunk.items():
                        if t_i not in cls.data:
                            cls.data[t_i] = data
                        else:
                            for h, val in data.items():
                                cls.data[t_i][h] = np.concatenate((cls.data[t_i][h], val))


            MPI.Finalize()
                        

        else:
            raise ValueError("No valid multiprocessing method selected.")
                
        # Return to original directory
        os.chdir(og_dir)
        
    def hdf5Write( cls , filename , working_dir ):
        """
        This method writes cls.data to an *.h5 file.

        Args:
            filename (string):  The name of the hdf5 file that will be written.

        """

        import h5py as h5

        with h5.File( filename , 'w' ) as f:
            for key , value in cls.data.items():
                f.create_dataset(key, data=value)

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



