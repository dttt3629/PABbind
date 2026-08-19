import pandas as pd
import numpy as np 
import os
# import torch
from Bio.PDB import PDBParser, NeighborSearch,PDBIO,Select
from Bio.PDB import Structure, Model, Chain
from rdkit import Chem
import pickle
import abnumber
from abnumber import Chain
# change to your path 
from joblib import delayed,Parallel
from tqdm import tqdm
import math
import MDAnalysis as mda

# useage
 
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
one_to_three = {three_to_one[i]:i for i in three_to_one}  
def find_seq(path,chainid):
    parser = PDBParser()

    pro = parser.get_structure('protein',path)
    k = []
    for i in chainid:
        k.extend(i)
    chain_res = {i:[] for i in k}
    chain_ind = {i:[] for i in k}
    for res in pro.get_residues():
        if res.parent.id not in k:
            continue
        else:
            if res.resname not in three_to_one.keys():
                continue
            else:
                chain_res[res.parent.id].append( three_to_one[res.resname])
                chain_ind[res.parent.id].append(res.id[1])
    return chain_res,chain_ind

def get_infor(i):
    path = i.split('_')
    chainid = path[1:]
    chainid[-1]= chainid[-1][:-4]
    w = find_seq('./pocket/'+i,chainid)
    p = find_seq('./pocketed/'+i,chainid)
    return path[0],chainid,w[0],w[1],p[0],p[1]


from esm.pretrained import load_model_and_alphabet_core
import torch
import pickle
def load_from_local(path,name):

    model_data = torch.load(path+name+'.pt',map_location="cpu")
    regression_data = torch.load(path+name+'-contact-regression.pt',map_location="cpu")
    return load_model_and_alphabet_core(name,model_data,regression_data)

def esm_enc_pre(p,src,dst,model,batch_converter):
    # with open(graph_path+p,'rb') as f:
    #     g = pickle.load(f)
    ch = p.split('_')[1:-1]
    ch.append(p.split('_')[-1].split('.')[0])
    f_seq,f_ind = find_seq(src+p.split('.')[0]+'.pdb',ch)
    p_seq,p_ind = find_seq(dst+p.split('.')[0]+'.pdb',ch)
    data = []
    for c in f_seq.keys():
        seq = ''
        for s in f_seq[c]:
            seq +=s 

        data.append((c,seq))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[30], return_contacts=True)
    token_representations = results["representations"][30]
    pocket_ind ={k:[] for k in f_seq.keys()}
    pocket_embed={k:[] for k in f_seq.keys()}
    for k in f_seq.keys():
        if f_ind[k][-1]>10000:
            f_ind[k] = list(np.array(f_ind[k])-10000)
    for k in f_seq.keys():
        for i in p_ind[k]: 
            pocket_ind[k].append(f_ind[k].index(i)+1)
    for i in range(token_representations.shape[0]):
        pocket_embed[batch_labels[i]]= token_representations[i,pocket_ind[batch_labels[i]]]
    # return 0 
    return pocket_embed
def esm_enc_ppi(p,src,dst,model,batch_converter):
    # with open(graph_path+p,'rb') as f:
    #     g = pickle.load(f)
    ch = p.split('_')[1:-1]
    ch.append(p.split('_')[-1].split('.')[0])
    f_seq = {}
    f_ind = {}
    for z in ch:
        out = find_seq(src+p.split('_')[0]+'_'+z+'.pdb',z)
        f_seq.update(out[0])
        f_ind.update(out[1])
    p_seq,p_ind = find_seq(dst+p.split('.')[0]+'.pdb',ch)
    data = []
    for c in f_seq.keys():
        seq = ''
        for s in f_seq[c]:
            seq +=s 

        data.append((c,seq))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[30], return_contacts=True)
    token_representations = results["representations"][30]
    pocket_ind ={k:[] for k in f_seq.keys()}
    pocket_embed={k:[] for k in f_seq.keys()}

    for k in f_seq.keys():
        for i in p_ind[k]: 
                pocket_ind[k].append(f_ind[k].index(i)+1)
    for i in range(token_representations.shape[0]):
        pocket_embed[batch_labels[i]]= token_representations[i,pocket_ind[batch_labels[i]]]
    # return 0 
    return pocket_embed
def esm_main():
    esmmodel, alphabet = load_from_local(path = "./ckpt/",name = 'esm2_t30_150M_UR50D')
    esmmodel.eval()
    path = os.listdir('./graph/sabgraph/')
    batch_converter = alphabet.get_batch_converter()
    src = './sab_clean/'
    dst = './pocket/'
    for k in tqdm(range(0,len(path))):
        i=path[k]
        if not os.path.exists('./graph/seqemb/'+i):
            # try:
                seq_emb = esm_enc_pre(i,src,dst,esmmodel,batch_converter)
                with open('./graph/seqemb/'+i,'wb')as f:
                    pickle.dump(seq_emb,f)
            # except:
            #     continue
def esm_main_ppi():
    esmmodel, alphabet = load_from_local(path = "./ckpt/",name = 'esm2_t30_150M_UR50D')
    esmmodel.eval()
    path = os.listdir('./graph/ppigraph/')
    batch_converter = alphabet.get_batch_converter()
    src = './01-benchmark_pdbs/'
    dst = './ppipock/'
    for i in tqdm(path):
        # if not os.path.exists('./graph/ppiseqemb/'+i):
            seq_emb = esm_enc_ppi(i,src,dst,esmmodel,batch_converter)
            with open('./graph/ppiseqemb/'+i,'wb')as f:
                pickle.dump(seq_emb,f)
def esm_main_ppi():
    esmmodel, alphabet = load_from_local(path = "./ckpt/",name = 'esm2_t30_150M_UR50D')
    esmmodel.eval()
    path = os.listdir('./graph/ppigraph/')
    batch_converter = alphabet.get_batch_converter()
    src = './01-benchmark_pdbs/'
    dst = './ppipock/'
    for i in tqdm(path):
        # if not os.path.exists('./graph/ppiseqemb/'+i):
            seq_emb = esm_enc_ppi(i,src,dst,esmmodel,batch_converter)
            with open('./graph/ppiseqemb/'+i,'wb')as f:
                pickle.dump(seq_emb,f)
def rep_check_updata(f_seq,f_ind,p_seq,p_ind):
    rep_dic={k:[] for k in f_seq.keys()}
    abls=['i','a','b','c','d','e','f','g','h','i','j','k','l','o','p','q','r','s','t']
    for k in f_seq.keys():
        for i in range(len(f_ind[k])-1):
            if f_ind[k][i]==f_ind[k][i+1]:
                rep_dic[k].append(f_ind[k][i])
        rep_dic[k]=list(set(rep_dic[k]))
    for k in f_seq.keys():
        zls = list(zip(f_seq[k],f_ind[k]))
        pls = list(zip(p_seq[k],p_ind[k]))
        for i in rep_dic[k]:
            zlssub = list(filter(lambda x:x[1]==i,zls))
            if i in p_ind[k]:
                plssub = list(filter(lambda x:x[1]==i,pls))
                if len(plssub)== len(zlssub):
                    ind = p_ind[k].index(i)
                    z = 1
                    for n in plssub[1:]:
                        p_ind[k][ind+z]=str(i)+abls[z]
                        z+=1
                else:
                    print('may wrong in',p)
                    ind = p_ind[k].index(i)
                    z = 0
                    # break
                    for n in plssub:
                        zind= zlssub.index(n)
                        if zind ==0:
                            continue
                        p_ind[k][ind+z]=str(i)+abls[zind]
                        z+=1
    # for k in f_seq.keys():
            zind = f_ind[k].index(i)
            z=1
            for n in zlssub[1:]:
                f_ind[k][zind+z] =str(i)+abls[z]
                z+=1
                    
def esm_enc_mut(p,ch,src,dst,model,batch_converter,muts):
    # with open(graph_path+p,'rb') as f:
    #     g = pickle.load(f)
    # ch = p.split('_')[1:-1]
    # ch.append(p.split('_')[-1].split('.')[0])
    f_seq,f_ind = find_seq(src+p+'.pdb',ch)
    p_seq,p_ind = find_seq(dst+p+'.pdb',ch)
    data = []
    rep_check_updata(f_seq,f_ind,p_seq,p_ind)
    for mut in muts:
        mut_ch = mut[0]
        wt_aa = mut[1]
        mt_aa = mut[-1]
        mt_ind = mut[2]
        # try:
        ind = f_ind[mut_ch].index(mt_ind)
        # except:
        #     ind = f_ind[mut_ch].index(int(mt_ind[:-1]))
        #     ind+= []
        assert wt_aa == f_seq[mut_ch][ind]
        f_seq[mut_ch][ind]=mt_aa
    for c in f_seq.keys():
        seq = ''
        for s in f_seq[c]:
            seq +=s 

        data.append((c,seq))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[30], return_contacts=True)
    token_representations = results["representations"][30]
    pocket_ind ={k:[] for k in f_seq.keys()}
    pocket_embed={k:[] for k in f_seq.keys()}
    for k in f_seq.keys():
        for i in p_ind[k]: 
            pocket_ind[k].append(f_ind[k].index(i)+1)
    for i in range(token_representations.shape[0]):
        pocket_embed[batch_labels[i]]= token_representations[i,pocket_ind[batch_labels[i]]]
    # return 0 
    return pocket_embed
def mut_esm_main():
        esmmodel, alphabet = load_from_local(path = "./ckpt/",name = 'esm2_t30_150M_UR50D')
        esmmodel.eval()
        src = './graph/dms/pdb/'
        dst = './graph/dms/pocket/'
        seq_emb =[] 
        batch_converter = alphabet.get_batch_converter()
        k= 0
        for i in tqdm(range(k,len(data))):
            p = data.iloc[i,0]
            ch = data.iloc[i,1].split('_')
            muts2 = data.Mutation[i].split(',')
            muts = []
            for mut in muts2:
                ind = mut[3:-1]
                try:
                    ind = int(ind)
                except:
                    0
                muts.append([mut[0],mut[2],ind,mut[-1]])
            path ='p+'.pdb'
            
            seq_emb.append([i,esm_enc_mut(path,ch,src,dst,esmmodel,batch_converter,muts)])
        esmmodel, alphabet = load_from_local(path = "./ckpt/",name = 'esm2_t30_150M_UR50D')
        esmmodel.eval()
        src = './graph/dms/pdb/'
        dst = './graph/dms/pocket/'
        seq_emb =[] 
        batch_converter = alphabet.get_batch_converter()
        k= 0
        for i in tqdm(range(k,len(data))):
            p2 = data.antibody[i]+'.pdb'
            p1 = data.crystal[i]+'.pdb'
            ch = [data.Hchain[i]+data.lchain[i],data.Rchain[i]]
            muts2 = []
            muts = []

            # path ='p+'.pdb'
            
            seq_emb.append([data.antibody[i],esm_enc_mut(p1,p2,ch,src,dst,esmmodel,batch_converter,muts)])
def pdb_emb(model,path):
    with open(path,'rb') as f:
        g = pickle.load(f)
    model.eval()
    g.node_s = g.node_s.to(torch.float32)
    g.node_v = g.node_v.to(torch.float32)
    g.edge_s = g.edge_s.to(torch.float32)
    g.edge_v = g.edge_v.to(torch.float32)
    g.segid = [g.segid]
    with torch.no_grad():
        enc = model(g)
    return enc
def pdb_emb_main():
    graphemb = []
    for i in range(len(data)):
        path = './graph/dms/graph/'+data.antibody[i]+'.pdb.bin'
        model = mlmmodel.inter_gvp
        mlmmodel.inter_gvp.gvp_model.issegemb = False
        with open(path,'rb') as f:
            g = pickle.load(f)

        graphemb.append([data.antibody[i],pdb_emb(model,path),np.unique(np.array(g.resid)[:,0])])
    with open('./graph/dms/mlm_emb.bin','wb') as f:
        pickle.dump(graphemb,f)
