from pymatgen.io.cif import CifWriter, CifParser
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
import torch
import numpy as np

## CIF -> (Array[[node_index,connected_node_index],...], [dist,...]) ##

#takes cif file and returns array (2 x num_edges) of edge index
#found by collecting neighbors within radius, also adjoins distance
#associated with each edge in tuple
def cif2graphedges(cif_file, radius:int=3):
    struc = CifParser(cif_file).get_structures()[0]
    nbr_lst = struc.get_neighbor_list(radius, exclude_self=True)
    edge_list=np.stack((nbr_lst[0],nbr_lst[1])).transpose()
    edge_list=torch.tensor(edge_list)
    edge_list_w_dist = (edge_list,torch.tensor(nbr_lst[3]))
    return edge_list_w_dist


## CIF -> tensor([[node_pos], ... ]) ##

#takes cif file and returns tensor of node positions indexed same as cif2graphedges
def cif2nodepos(cif_file):
    struc = CifParser(cif_file).get_structures()[0]
    site_lst = struc.sites
    index = 0
    nodepos_lst = []
    nodespec_lst = []
    for site in site_lst:
        nodepos_lst.append(site.coords)
        nodespec_lst.append(site.species)
    nodepos_arr = np.array(nodepos_lst, dtype=float)
    return  (torch.tensor(nodepos_arr),nodespec_lst)


## CIF file -> torch_geometric graph ##

#takes cif file (and optional radius for neighbor search)
#as input and returns torch_geometric graph with distances as edge_attr
#node positions from cif file and graph edges for all neighbors within radius
def cif2graph(cif_file, radius:int=3):
    pos=cif2nodepos(cif_file)[0]
    x=cif2nodepos(cif_file)[1] #Still in strings of elements, consider conv to values
    edge_index=cif2graphedges(cif_file, radius=radius)[0]
    edge_attr=cif2graphedges(cif_file, radius=radius)[1]
    cgraph=Data(pos=pos, edge_index=edge_index, edge_attr=edge_attr )
    return cgraph

