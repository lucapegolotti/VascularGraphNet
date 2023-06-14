import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tools.io_tools import collect_arrays, collect_points, save_vtk, read_geo
import dgl
import torch as th

def simplify_and_collect(solution, boundary, sampling_rate, nneighbors):
    """
    Select nodes in the mesh and collect relevant arrays.
  
    Arguments:
        solution (vtkXMLUnstructuredGridReader): Solution
        boundary (vtkPolyData): Boundary data containing ModelFaceID
        sampling_rate: rate used to undersample nodes in the mesh
        nneighbors: number of closest neighbors for graph construction
    Returns:
        points: 2D array containing points of the graph
        edges: tuple containing two arrays with first nodes and second nodes in
               indices
        fields: dictionary containing node and edge features

    """
    points = collect_points(solution.GetOutput().GetPoints())
    N = points.shape[0]
    M = int(N*sampling_rate)
    random_indices = np.random.choice(N, size=M, replace=False)

    # Select the random rows from the matrix
    points = points[random_indices, :]

    k = nneighbors + 1
    nn = NearestNeighbors(n_neighbors=k)

    nn.fit(points)
    _, indices = nn.kneighbors(points)
    
    edges = set()
    for i in range(M):
        # the first closest neighbor is itself
        for j in range(1, k):
            edges.add((i, indices[i][j]))
            edges.add((indices[i][j], i))

    e1 = []
    e2 = []
    for edge in edges:
        e1.append(edge[0])
        e2.append(edge[1])

    arrays = collect_arrays(solution.GetOutput().GetPointData())
    ntimes = 0
    velocities = {}
    pressures = {}
    for array_name, array in arrays.items():
        if 'average' not in array_name:
            if 'pressure' in array_name:
                pressures[array_name] = array[random_indices] / 1333.2
            if 'velocity' in array_name:
                velocities[array_name] = array[random_indices]
                if 'velocity' in array_name:
                    ntimes = ntimes + 1

    velocity = np.zeros((points.shape[0], 3, ntimes))
    pressure = np.zeros((points.shape[0], ntimes))

    itime = 0
    for v in velocities:
        velocity[:,:,itime] = velocities[v]
        itime = itime + 1

    itime = 0
    for p in pressures:
        pressure[:,itime] = pressures[p]
        itime = itime + 1

    points_boundary = collect_points(boundary.GetOutput().GetPoints())
    point_map_boundary = {}
    for i, point in enumerate(points_boundary):
        diff = np.linalg.norm(points - point, axis = 1)
        point_map_boundary[i] = np.argmin(diff)

    arrays_boundary = collect_arrays(boundary.GetOutput().GetCellData())
    boundary_id = arrays_boundary['ModelFaceID']
    num_boundary_cells = boundary.GetOutput().GetNumberOfCells()

    ids = np.zeros((points.shape[0]))
    for i in range(num_boundary_cells):
        cur_id = boundary_id[i]
        # we don't distinguish among outlets (id >= 3)
        if cur_id >= 3:
            cur_id = 3
        
        cell = boundary.GetOutput().GetCell(i)
        points_in_cell = cell.GetPointIds()

        point_ids = []
        for j in range(points_in_cell.GetNumberOfIds()):
            point_ids.append(points_in_cell.GetId(j))

        npoints = len(point_ids)
        for j in range(npoints):
            ids[point_map_boundary[point_ids[j]]] = cur_id

    edge_features = np.zeros((len(e1),4))

    for i in range(len(e1)):
        diff = points[e2[i],:] - points[e1[i],:]
        ndiff = np.linalg.norm(diff)
        edge_features[i,0:3] = diff / ndiff
        edge_features[i,3] = ndiff

    fields = {'velocity': velocity,
              'pressure': pressure,
              'id': ids.astype(np.int64),
              'edge_features': edge_features}
    
    print('Mesh nodes before reduction = {:}'.format(N))
    print('Graph nodes after reduction = {:}'.format(M))
    print('Graph edges after reduction = {:}'.format(len(e1)))

    return points, (e1, e2), fields

def generate_dgl_graph(points, edges, fields):
    graph = dgl.graph((edges[0], edges[1]), idtype = th.int32)

    graph.ndata['x'] = th.tensor(points, dtype = th.float32)
    graph.ndata['velocity'] = th.tensor(fields['velocity'], dtype=th.float32)
    graph.ndata['pressure'] = th.tensor(fields['pressure'], dtype=th.float32)
    graph.ndata['id'] = th.nn.functional.one_hot(th.tensor(fields['id']),
                                                 num_classes = 4)
    graph.edata['edge_features'] = th.tensor(fields['edge_features'], 
                                             dtype=th.float32)
    return graph

if __name__ == "__main__":
    sol3Dfile = 'test_data/solution.vtu'
    sol3D = read_geo(sol3Dfile)

    boundary_file = 'test_data/boundary.vtp'
    boundary = read_geo(boundary_file)

    points, edges, fields = simplify_and_collect(sol3D, boundary, 
                                                 sampling_rate = 0.5,
                                                 nneighbors=8)
    
    graph = generate_dgl_graph(points, edges, fields)
    dgl.save_graphs('graph.grph', graph)
    # save_vtk('graph.vtk', graph)