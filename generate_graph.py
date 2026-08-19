from itertools import chain
from feature.protein_feature import get_protein_feature_mda,get_protein_feature_se3
from joblib import delayed,Parallel
from tqdm import tqdm
import argparse
import pickle as pkl
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import os 
import pdb

def prot_generate_graph(path,chain_id=[]):
    result = get_protein_feature_mda(path,chain_id= chain_id)
    p_xyz, _, p_seq, p_node_s, p_node_v, p_edge_index, p_edge_s, p_edge_v, _ ,segid,resid= result
    g = Data(node_s = p_node_s,
            node_v = p_node_v,
            edge_s = p_edge_s,
            edge_v = p_edge_v,
            edge_index= p_edge_index,
            pos = p_xyz,
            seq = p_seq,
            segid = segid,
            resid = resid
                )
    return g
def get_ppi_graph(path,pdb_path):
    # path =1A22_AC_BD.pdb
    #usage 
    # Parallel(n_jobs = 10)(delayed(get_ppi_graph)(i)for i in pdbs )
    # atb_path = f'{path}_antibody.pdb'
    # atg_path = f'{path}_antigen.pdb'
    chain_id = path.split('.')[0].split('_')[1:]
    chain_id=[chain_id[1][0],chain_id[1][1],chain_id[0][0]]
    g = prot_generate_graph(path = pdb_path+'pocket/'+path,chain_id=chain_id)
    with open(f'{pdb_path+'graph/'+path.split(".")[0]}.bin','wb') as f:
        pkl.dump(g,f)
