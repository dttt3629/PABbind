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

def get_graph(path,chain_id):
    # atb_path = f'{path}_antibody.pdb'
    # atg_path = f'{path}_antigen.pdb'
    g = prot_generate_graph(path = './pocket/'+path,chain_id=chain_id[1:])
    with open(f'./graph/sabgraph/{path}.bin','wb') as f:
        pkl.dump(g,f)
    # with open(f'{dst_path}/{path}_antigen.bin','wb') as f:
    #     pkl.dump(prot_generate_graph(f'{src_path}/{path}_antigen.pdb'),f)
def get_ppi_graph(path):
    # path =1A22_AC_BD.pdb
    #usage 
    # Parallel(n_jobs = 10)(delayed(get_ppi_graph)(i)for i in pdbs )
    # atb_path = f'{path}_antibody.pdb'
    # atg_path = f'{path}_antigen.pdb'
    chain_id = path.split('_')[1:]
    chain_id.append( chain_id[1][:-4])
    chain_id[1]=''
    g = prot_generate_graph(path = './ppipock/'+path,chain_id=chain_id)
    with open(f'./graph/ppigraph/{path}.bin','wb') as f:
        pkl.dump(g,f)
def parallel_generate_graph(data,receptor):
    for rec in receptor:
        if not os.path.exists(f'{dst_path}/{rec}'):
            os.mkdir(f'{dst_path}/{rec}')
    for rec in receptor:
        sele_data = data[data.iloc[:,0]==rec]
        sele_data = sele_data.iloc[:-1,:]
        paths= [f'{rec}/min4-{sele_data.iloc[i,1]}' for i in range(len(sele_data))]
        graphs = Parallel(n_jobs = -1)(delayed(get_graph)(paths[i]) for i in tqdm(range(len(paths))))
        sele_data.to_csv(f'./data/{rec}.csv',index=False)
        atb, atg = list(zip(*graphs))
        with open(f'{dst_path}/{rec}_antibody.bin','wb') as f:
            pkl.dump(atb,f)
        with open(f'{dst_path}/{rec}_antigen.bin','wb') as f:
            pkl.dump(atg,f)
if __name__ == '__main__':
    # p = argparse.ArgumentParser()
    # p.add_argument('--src_path', type=str, default='./data/pocket')
    # p.add_argument('--dst_path', type=str, default='./graph')
    # p.add_argument('--data_path', type=str, default='./data.csv')
    # args = p.parse_args()

    # data = pd.read_csv(args.data_path,header=None)
    # receptor =list(data.iloc[:,0].drop_duplicates())
    # src_path = args.src_path
    # dst_path = args.dst_path
    # parallel_generate_graph(data,receptor)
    path = os.listdir('./pocket')
    chainls = []
    for i in path:
        s = i.split('_')
        temp = []
        # temp.append(s[0])
        for c,d in enumerate(s):
            if c == 0:
                temp.append(d)
            else:
                if len(d)==1:
                    temp.append(d)
                else:
                    temp.append(d[0])
        chainls.append(temp)
    Parallel(n_jobs = 10)(delayed(get_graph)(path[i],chainls[i]) for i in tqdm(range(len(path))))
    k=0
    for i in tqdm(range(k,len(path))):
        if os.path.exists(f'./graph/sabgraph/{path[i]}.bin'):
            continue
        # else:
            # try:
        get_graph(path[i],chainls[i])
            # except:
            #     pdb.set_trace()
        with open('sab_chainls.pkl','wb') as f:
            pkl.dump(chainls,f)