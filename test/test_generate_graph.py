import sys
import os
sys.path.append(os.getcwd())

import dgl
from graph.process import simplify_and_collect, generate_dgl_graph
from tools.io_tools import read_geo, save_vtk

if __name__ == "__main__":
    sol3Dfile = 'test/test_data/solution.vtu'
    sol3D = read_geo(sol3Dfile)

    boundary_file = 'test/test_data/boundary.vtp'
    boundary = read_geo(boundary_file)

    points, edges, fields = simplify_and_collect(sol3D, boundary, 
                                                 sampling_rate = 0.5,
                                                 nneighbors=8)
    
    graph = generate_dgl_graph(points, edges, fields)
    dgl.save_graphs('graph.grph', graph)
    save_vtk('graph.vtk', graph)