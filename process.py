


import math
import logging
import os
from scipy.stats import truncnorm
from tqdm import tqdm
import numpy as np
from model.test_model import myconfig, Inter_gvp_pre, Inter_gvp,atbbert
import torch
import re
import pandas as pd
from torch.nn import functional as F
import logging
import argparse
import random
import pickle
from torch import nn
import torch.multiprocessing as mp
from esm.pretrained import load_model_and_alphabet_core
import json
import pdb
import copy
from utils.extract_pocket import run_get_ppi_pocket
from utils.utils import get_imgt_str
from joblib import delayed,Parallel
from generate_graph import get_ppi_graph
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def load_from_local(path,name):

    model_data = torch.load(path+name+'.pt',map_location="cpu" ,weights_only=False)
    regression_data = torch.load(path+name+'-contact-regression.pt',map_location="cpu", weights_only=False)
    return load_model_and_alphabet_core(name,model_data,regression_data)
def pdb_emb(model,g):

    model.eval()
    g.node_s = g.node_s.to(torch.float32)
    g.node_v = g.node_v.to(torch.float32)
    g.edge_s = g.edge_s.to(torch.float32)
    g.edge_v = g.edge_v.to(torch.float32)
    g.segid = [g.segid]
    with torch.no_grad():
        enc = model(g)
    return enc
def parallel_get_ppi_pocket(i,Rchain,Lchain,Hchain,crys,pdb_path):
    rchain = Rchain[i]
    lchain = Lchain[i]
    hchain = Hchain[i]
    cry = crys[i].strip()
    inputs = f'{cry}_{rchain}_{hchain}_{lchain}'
    run_get_ppi_pocket(inputs,pdb_path)
from esm.pretrained import load_model_and_alphabet_core
import torch
import pickle
def load_from_local(path,name):

    model_data = torch.load(path+name+'.pt',map_location="cpu" ,weights_only=False)
    regression_data = torch.load(path+name+'-contact-regression.pt',map_location="cpu", weights_only=False)
    return load_model_and_alphabet_core(name,model_data,regression_data)
esmmodel, alphabet = load_from_local(path = "/home/dengqr/esm/",name = 'esm2_t30_150M_UR50D')
esmmodel.eval()
class ChunkedIterator:
    def __init__(self, data, chunk_size=4):
        self.data = data
        self.chunk_size = chunk_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.data):
            # 计算当前块的结束索引
            end_index = min(self.index + self.chunk_size, len(self.data))
            # 获取当前块的数据
            chunk = self.data[self.index:end_index]
            # 更新索引
            self.index = end_index
            return chunk
        else:
            raise StopIteration
def enc(model,alphabet,data,length=None):
    t=None
    if length != None:
        data.append(('test',''.join(['G']*length)))
        t=-1
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    model.cuda()
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[30], return_contacts=True)
    token_representations = results["representations"][30]
    return token_representations[:t,:].cpu()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./abdataset/dmscheckclean.csv')
    parser.add_argument('--pdb_path', type=str, default='./graph/dms/')
    parser.add_argument('--esm_path', type=str, default='./esm2_t30_150M_UR50D/')
    parser.add_argument('--mt_path', type=str, default='./abdataset/total_dup3.csv')
    args = parser.parse_args()
    set_seed(42)
    config_log = json.load(open('./config/train_mlmgp_lg_tg.json','r'))
    data = pd.read_csv(args.data_path)
    if not os.path.exists(args.pdb_path+'pdb'):
        raise FileNotFoundError('pdb folder not found')
    # get pocket
    Rchain = data.Rchain.to_list()
    Lchain = data.lchain.to_list()
    Hchain = data.Hchain.to_list()
    crys = data.crystal.to_list()
    Parallel(n_jobs= 10)(delayed(parallel_get_ppi_pocket)(i,Rchain,Lchain,Hchain,crys,args.pdb_path)for i in range(len(Rchain)))
    # get graph input
    inputs =  [f'{crys[i]}_{Rchain[i]}_{Hchain[i]}_{Lchain[i]}' for i in range(len(Rchain))]
    Parallel(n_jobs= 10)(delayed(get_ppi_graph)(i,args.pdb_path) for i in inputs)
    # get graph emb
    with open('./config/train_mlmgp_lg_pred.json','r') as f:
    strf = f.read()
    config_log = json.loads(strf)
    batch_converter = alphabet.get_batch_converter()
    model_config = myconfig(configdic=config_log['model_config'])
    graph_inter_model = Inter_gvp(model_config)
    graph_model = Inter_gvp_pre(model_config,graph_inter_model)
    graph_model.load_state_dict(torch.load('./ckpt/'+model_config.graphmodel_path+'.pt',map_location='cpu',weights_only=False))
    graph_model.eval()
    graph_inter_model.eval()
    graphemb = []
    graph_model.inter_gvp.gvp_model.issegemb = False
    paths = os.listdir(args.pdb_path+'graph/')
    for p in paths:
        crys = p.split('.')[0]
        crystal =crys.split('_')[0]
        rchain = crys.split('_')[1]
        hchain, lchain = crys.split('_')[2]
        antibody = data[(data['crystal']==crystal)&(data['Rchain']==rchain)&(data['Hchain']==hchain)&(data['lchain']==lchain)].antibody.values[0]
        path = args.pdb_path+'graph/'+p
        with open(path,'rb') as f:
            g = pickle.load(f)
        graphemb.append([antibody,pdb_emb(graph_inter_model,g),np.unique(np.array(g.resid)[:,0])])
    with open(args.pdb_path+'mlm_emb.bin','wb') as f:
            pickle.dump(graphemb,f)
    # wtesm emb
    esmmodel, alphabet = load_from_local(path = args.esm_path,name = 'esm2_t30_150M_UR50D')
    esmmodel.eval()
    data = pd.read_csv(args.data_path)
    paths = os.listdir(args.pdb_path+'pocket/')
    src = args.pdb_path+'pdbclean/'
    dst = args.pdb_path+'pocket/'
    seq_emb =[] 
    batch_converter = alphabet.get_batch_converter()
    for p in paths:
        crys = p.split('.')[0]
        crystal =crys.split('_')[0]
        rchain = crys.split('_')[1]
        hchain, lchain = crys.split('_')[2]
        ch = [hchain+lchain,rchain]
        p1 = crystal
        p2 = crys
        muts2 = []
        muts = []
        antibody = data[(data['crystal']==crystal)&(data['Rchain']==rchain)&(data['Hchain']==hchain)&(data['lchain']==lchain)].antibody.values[0]
        seq_emb.append([antibody,esm_enc_mut(p1,p2,ch,src,dst,esmmodel,batch_converter,muts)])
        with open(args.pdb_path+'wtseq_emb.bin','wb') as f:
            pickle.dump(seq_emb,f)
    # mt emb
    subdata1 = pd.read_csv(args.mt_path)
    sites = (subdata1.site).tolist()
    wild = (subdata1.wildtype).tolist()
    mutation = (subdata1.mutation).tolist()
    mut = np.unique([wild[i]+str(sites[i]-330)+mutation[i] for i in range(len(sites))])
    to_add = []
    for m in mut:
        if m not in muts:
            to_add.append(m)
    sequences = []
    Rseq ='NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

    for m in to_add:
        ind = int(m[1:-1])-1
        assert m[0]==Rseq[ind]
        
        sequences.append(Rseq[:ind]+m[-1]+Rseq[ind+1:])
    seqgen = ChunkedIterator(sequences)
    out = torch.tensor([])
    for s in seqgen:
        k  = list(range(len(s)))
        data = list(zip(k,s))
        out = torch.concat([out,enc(esmmodel,alphabet,data)])
    record = []
    for i in range(len(to_add)):
        record.append((to_add[i],sequences[i]))
    with open('esm_embd.pkl','wb')as f:
        pickle.dump((record,out),f)
