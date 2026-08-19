import sre_compile
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
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
one_to_three = {three_to_one[i]:i for i in three_to_one}
def get_ppi_pocket(path,keep_chain,src_dic,save_dic,cutoff=10.0):
    #keep the max antigen chain in 10 ai
    # keep_chain [A,B]
    #usage

        
    parser = PDBParser()

    pro = parser.get_structure('protein',src_dic+path)
    chains =  [ch for ch in pro.get_chains()]
    chainid =[]
    chain_res = {}
    for ch in chains:
        # chainid.append(keep_chain.index(ch.id))
        chain_res[ch.id] = [res for res in ch.get_residues()]
    
    inter_res = {i:[] for i in chain_res.keys()}

    for i,chid in enumerate(chain_res.keys()):
        if chid in keep_chain[1]:

            # print(i,chid)
            
            # keep_resH = []
            keep_res = {i:[] for i in chain_res.keys()}
            # keep_resB = {i:[] for i in keep_chain[1]}
            ch_res = chain_res[chid]
            for c in ch_res:
                # try:
                    flag = False
                    for ch in keep_chain[0]:
                        for h in chain_res[ch]:
                            if math.dist(c.child_dict['CA'].coord,h.child_dict['CA'].coord) < cutoff:
                                if h in keep_res[ch]:
                                    continue
                                else:
                                    keep_res[ch].append(h.get_id()[1])
                                flag = True

                    if flag:
                        keep_res[chid].append(c.get_id()[1])
                # except Exception as e:
                #     print(path)
                #     continue
            for k in chain_res.keys():
                inter_res[k].extend(keep_res[k]) 

    for k in inter_res.keys():
        inter_res[k]= list(sorted(set(inter_res[k])))
    total_num = sum([len(i) for i in inter_res])
    def closer_fill(resls,ch_res,fillgap = 10,long_cut=10):
        # to ensure the continuity
        # input sorted resls id
        insert=[]
        for i in range(0,len(resls)-1):
            if abs(resls[i] -resls[i+1])!=1 and abs(resls[i] -resls[i+1]) <=fillgap :
                insert.extend(range(resls[i]+1,resls[i+1]))
        # ensure the lacked res
        for i in ch_res:
            if i.get_id()[1]  in insert:
                resls.append(i.get_id()[1])
        resls = sorted(resls)
        #remove single res
        sgres = []
        for i in range(0,len(resls)-1):
                if abs(resls[i] -resls[i+1])>long_cut:
                    if i==0:
                        #strict for first
                        if abs(resls[i] -resls[i+1])>fillgap:
                            sgres.append(i)
 
                    elif i == len(resls)-1:
                        sgres.append(i+1)
                    elif abs(resls[i] -resls[i-1])>long_cut:
                        sgres.append(i)
                        
                    else:
                        continue
        resls = [i for i in resls if resls.index(i) not in sgres]
        return resls
    for i in inter_res.keys():
        inter_res[i] = closer_fill(inter_res[i],chain_res[i])
    class inter_res_select(Select):
        def accept_residue(self, res):
            chain_id = res.get_parent().id

            if res.get_id()[1] in inter_res[chain_id]:
                return True
            else:
                return False
    io = PDBIO()
    io.set_structure(pro)
    path = path.split('.')[0]
    io.save(f'{save_dic}/{path}_{keep_chain[0]}_{keep_chain[1]}.pdb',inter_res_select())
def run_get_ppi_pocket(inputs,pdb_path):
        #usage
        #Parallel(n_jobs= 10)(delayed(run_get_ppi_pocket)(i)for i in pdbs)
        # for i in pdbs: # ['1A22_B_A']
        # if i not in os.listdir('./pocketed'):
            p=inputs.split('_')
            path = p[0]+'.pdb'
            keep_chain = [p[1],''.join(p[2:])]
            src_dic = pdb_path+'pdb/'
            save_dic = pdb_path+'pocket/'
            get_ppi_pocket(path,keep_chain,src_dic,save_dic)

