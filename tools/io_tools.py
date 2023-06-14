import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
import os 
import numpy as np
import meshio

def read_geo(fname):
    """
    Read geometry from file.
  
    Arguments:
        fname: File name
    Returns:
        The vtk reader
  
    """
    _, ext = os.path.splitext(fname)
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError("File extension " + ext + " unknown.")
    reader.SetFileName(fname)
    reader.Update()
    return reader

def collect_points(celldata, components = None):
    """
    Collect points from a cell data object.
  
    Arguments:
        celldata: Name of the directory
        components (int): Number of array components to keep. 
                          Default: None -> keep allNone
    Returns:
        The array of points (numpy array)
  
    """
    if components == None:
        res = v2n(celldata.GetData()).astype(np.float32)
    else:
        res = v2n(celldata.GetData())[:components].astype(np.float32)
    return res

def collect_arrays(celldata, components = None):
    """  
    Collect arrays from a cell data or point data object.
  
    Arguments:
        celldata: Input data
        components (int): Number of array components to keep. 
                          Default: None -> keep all
    Returns:
        A dictionary of arrays (key: array name, value: numpy array)
  
    """
    res = {}
    for i in range(celldata.GetNumberOfArrays()):
        name = celldata.GetArrayName(i)
        data = celldata.GetArray(i)
        if components == None:
            res[name] = v2n(data).astype(np.float32)
        else:
            res[name] = v2n(data)[:components].astype(np.float32)
    return res

def get_all_arrays(geo, components = None):
    """
    Get arrays from geometry file.
  
    Arguments:
        geo: Input geometry
        components (int): Number of array components to keep. 
                          Default: None -> keep all
    Returns:
        Point data dictionary (key: array name, value: numpy array)
        Cell data dictionary (key: array name, value: numpy array)
        Points (numpy array)
  
    """
    # collect all arrays
    cell_data = collect_arrays(geo.GetCellData(), components)
    point_data = collect_arrays(geo.GetPointData(), components)
    points = collect_points(geo.GetPoints(), components)
    return point_data, cell_data, points

def save_vtk(outfile, graph):

    points = graph.ndata['x']

    edges0 = graph.edges()[0]
    edges1 = graph.edges()[1]

    cells = {
        'line': np.vstack((edges0, edges1)).transpose()
    }

    meshio.write_points_cells(outfile, points, cells)