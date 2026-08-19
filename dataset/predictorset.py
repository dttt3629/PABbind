import numpy as np 
import pandas as pd 
import torch
from torch import nn
from torch.fx import graph
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F
import math 
import scipy
import copy
import pickle as pkl
def esm_enc(p,mut,model,batch_converter,embed_chain ):
    _,_,f_seq,f_ind,p_seq,p_ind =copy.deepcopy(p[0])
    mut_ch = mut[0]
    wt_aa = mut[1]
    mt_aa = mut[-1]
    mt_ind = mut[2]
    ind = p[0][3][mut_ch].index(mt_ind)
    f_seq[mut_ch][ind]=mt_aa
    
    data = []
    for c in embed_chain:
        seq = ''
        for s in f_seq[c]:
            seq +=s 

        data.append((c,seq))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[30], return_contacts=True)
    token_representations = results["representations"][30]
    pocket_ind ={k:[] for k in f_seq.keys()}
    pocket_embed={k:[] for k in f_seq.keys()}
    for k in embed_chain:
        for i in p_ind[k]:
            pocket_ind[k].append(f_ind[k].index(i)+1)
    for i in range(token_representations.shape[0]):
        pocket_embed[batch_labels[i]]= token_representations[i,pocket_ind[batch_labels[i]]]
    # return 0 
    return pocket_embed
def seq_enc_prepare(pdbls,pdbs,mut):
    for p in pdbls:
        if p[0][0]==pdbs:
            break
    mut = mut.split(':')
    mutation = [mut[0]]
    mutation.extend([mut[1][0],mut[1][1:-1],mut[1][-1]])
    return p,mutation
class ppi_dataset(Dataset):
    # MAX_PPI =461
    def __init__(self,set_up,seq_emb,graph_emb,wtseq_emb):
    #data [pdbid,mutation]
        super().__init__()
        # self.pdbid,self.mutation = data
        # self.pdbls = pdbls
        self.graph_emb = graph_emb
        self.set_up = set_up
        self.seq_emb = seq_emb
        self.wtseq_emb = wtseq_emb
        # self.muts = np.array(muts)
        
    def __len__(self):
        return len(self.seq_emb)
    def __getitem__(self, index):
        label = torch.tensor(float(self.set_up[index][-1]))
        muts = self.set_up[index][1]
        pdbs = self.set_up[index][0]
        for p in self.graph_emb:
            if p[0]==pdbs:
                break
        for wt in self.wtseq_emb:
            if wt[0]==pdbs:
                break
        order = p[-1]
        wt_emb = torch.concat([wt[1][k] for k in order])
        graph_emb =  p[1]
        mt_emb = torch.concat([self.seq_emb[index][1][k] for k in order])
        valid_len=wt_emb.shape[0]
        # wt_emb = torch.concat([wt_emb,torch.zeros(self.MAX_PPI-valid_len,wt_emb.shape[1])])
        # mt_emb = torch.concat([mt_emb,torch.zeros(self.MAX_PPI-valid_len,mt_emb.shape[1])])
        # graph_emb = torch.concat([graph_emb,torch.zeros(self.MAX_PPI-valid_len,graph_emb.shape[1])])
        # mask = torch.zeros(self.MAX_PPI,1)
        # mask[:valid_len,0]=1
        return wt_emb,mt_emb,graph_emb,label,muts,valid_len
def ppi_collate_fn(batch):
    wt,mt,gp,lb,mut,vl = zip(*batch)
    max_length = max(len(item) for item in wt)
    masks = []
    wt2 = []
    mt2=[]
    gp2 = []
    for i in range(len(wt)):
        wt2.append( torch.concat([wt[i],torch.zeros(max_length-vl[i],wt[i].shape[1])]))
        mt2.append( torch.concat([mt[i],torch.zeros(max_length-vl[i],mt[i].shape[1])]))
        gp2.append( torch.concat([gp[i],torch.zeros(max_length-vl[i],gp[i].shape[1])]))
        mask = torch.zeros(max_length,1)
        mask[:vl[i],0]=1
        masks.append(mask) 
    return torch.stack(wt2),torch.stack(mt2),torch.stack(gp2),torch.stack(lb),torch.stack(masks),vl

class dms_dataset(Dataset):
    # MAX_PPI =461
    def __init__(self,config,data,esm_emb,graph_emb,wtseq_emb,data2,antbdic):
    #data [pdbid,mutation]
        super().__init__()
        # self.pdbid,self.mutation = data
        # self.pdbls = pdbls
        self.config = config
        self.esm_emb = esm_emb
        self.graph_emb = graph_emb
        self.data = data
        self.wtseq_emb = wtseq_emb
        self.data2 = data2
        self.antbdic = antbdic
        self.antb = self.data.iloc[:,1]
        # self.muts = np.array(muts)
        self.muts = self.data.iloc[:,3]
        self.label = self.data.iloc[:,2]
        self.mutindex = []
        muts,_=zip(*esm_emb[0])
        for m in self.muts:
            self.mutindex.append(muts.index(m))
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        antbdic = self.antbdic
        antb = self.antb[index]
        label = self.label[index]
        muts = self.muts[index]
        Rchain = antbdic[antb].split('_')[1]
        for p in self.graph_emb:
            if p[0]==antb:
                break

        for wt in self.wtseq_emb:
            if wt[0]==antb:
                break
        graph_path = self.config["graph_path"]
        with open(f'{graph_path}{antbdic[antb]}.bin','rb') as f:
            g= pkl.load(f)
        p_ind = list(filter(lambda x:x[0]==Rchain,g.resid))
        _,pid = zip(*p_ind)
        order = p[-1]
        pid = np.array(pid)-330 
        wt_emb = torch.concat([wt[1][k] for k in order])
        mt = copy.deepcopy(wt)
        mt[1][Rchain] = self.esm_emb[1][self.mutindex[index]][pid,:]

        graph_emb =  p[1]
        mt_emb = torch.concat([mt[1][k]for k in order])
        valid_len=wt_emb.shape[0]

        return wt_emb,mt_emb,graph_emb,torch.tensor(label),muts,valid_len
def ppi_collate_fn(batch):
    wt,mt,gp,lb,mut,vl = zip(*batch)
    max_length = max(len(item) for item in wt)
    masks = []
    wt2 = []
    mt2=[]
    gp2 = []
    for i in range(len(wt)):
        wt2.append( torch.concat([wt[i],torch.zeros(max_length-vl[i],wt[i].shape[1])]))
        mt2.append( torch.concat([mt[i],torch.zeros(max_length-vl[i],mt[i].shape[1])]))
        gp2.append( torch.concat([gp[i],torch.zeros(max_length-vl[i],gp[i].shape[1])]))
        mask = torch.zeros(max_length,1)
        mask[:vl[i],0]=1
        masks.append(mask) 
    return torch.stack(wt2),torch.stack(mt2),torch.stack(gp2),torch.stack(lb),torch.stack(masks),vl
