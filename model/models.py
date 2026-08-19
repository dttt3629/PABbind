import numpy as np 
import pandas as pd 
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F
from model.GVP_Block import GVP_embedding
from torch_scatter import scatter_add,scatter_mean
from torch_geometric.utils import to_dense_batch
import math 
import scipy
import time
import pdb
class myconfig:
    def __init__(self, n_head=8, n_embd=512, attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop = 0.1,configdic={},**kwargs):
            self.n_embd = n_embd
            self.attn_pdrop = attn_pdrop
            self.resid_pdrop = resid_pdrop
            self.embd_pdrop = embd_pdrop
            self.n_head = n_head
            self.vocab_size= 26

            self.n_layer = 12
            for k,v in configdic.items():
                setattr(self, k, v)
            for k,v in kwargs.items():
                setattr(self, k, v)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        # martrix=torch.tril(torch.ones(11,11).view(1,1,11,11))
        # self.register_buffer("mask", martrix)
    def forward(self, x,mask=None):
        B, T, C = x.size()
        global time
        k= self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            # pdb.set_trace()
            att = att.masked_fill(mask == 1, float(-1e4))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y, attn_save

class Multhead_attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        # from my other job so just trans it
        hidden_dim = config.n_embd
        input_dim = config.n_embd
        head = config.n_head
        dropout_rate = config.attn_pdrop
        self.key = nn.Linear(input_dim,hidden_dim,bias=False)
        self.query = nn.Linear(input_dim,hidden_dim,bias=False)
        self.value = nn.Linear(input_dim,hidden_dim,bias=False)
        self.n_head = head
        self.pos = nn.Linear(3,hidden_dim,bias=False)
        # self.reset_parameters()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.resid_drop = nn.Dropout(dropout_rate)
        self.norm1 =nn.LayerNorm(hidden_dim)
        self.norm2 =nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), 
                nn.ELU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.Dropout(p=dropout_rate))
    def forward(self,Q,K,k_mask=None):
        B,T,C =K.shape  # receptor
        B,T2,C2 = Q.shape
        k= self.key(self.norm1(K)).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(self.norm2(Q)).view(B, T2, self.n_head, C2 // self.n_head).transpose(1, 2) # (B, nh, T2, hs)
        v = self.value(self.norm1(K)).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # mask = (q_mask.unsqueeze(-1))*(k_mask.unsqueeze(-2)) #(B,T2,T1)


        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))#(B,nh,T2,T1)
        if k_mask !=None:
            mask = k_mask.squeeze().unsqueeze(-2).repeat(1,T2,1)
            mask = mask.unsqueeze(1).repeat(1,self.n_head,1,1)
            att = att.masked_fill(mask[:,:,:,:] == 0, float(-1e4))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        # pdb.set_trace()
        y = att @ v # (B, nh, T2, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = torch.nan_to_num(y,1e-10)
        y = y.transpose(1, 2).contiguous().view(B, T2, C) # re-assemble all head outputs side by side
        # pdb.set_trace()
        y = self.proj(y)
        # pdb.set_trace()
        y = self.mlp(self.norm3(y))+Q
        return y, attn_save

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    def forward(self, x,mask=None):
        y, attn = self.attn(self.ln1(x),mask)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

class MaskLM(nn.Module):
    def __init__(self,config, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, config.n_embd),
                                 nn.ReLU(),
                                 nn.LayerNorm(config.n_embd),
                                 nn.Linear(config.n_embd, config.vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        batch_idx = torch.tensor(batch_idx).repeat_interleave( num_pred_positions)
        masked_X = X[batch_idx.to(torch.long), pred_positions.to(torch.long)]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
class NextSentencePred(nn.Module):
    def __init__(self, config, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(config.n_embd, 2)

    def forward(self, X):
        return self.output(X)




class Inter_gvp(nn.Module):
    # gvp for encode the res 
    # mate for rest param
    # mlp
    def __init__(self,config, dropout_rate=0.15,
                    dist_threhold=1000):
        super().__init__()
        self.config = config
        in_channels = config.n_embd
        hidden_dim = config.n_embd
        
        self.gvp_model = GVP_embedding(node_in_dim = (config.node_s, config.node_v),
                                    node_h_dim = (config.n_embd,16),
                                    edge_in_dim = (config.edge_s, 1),
                                    edge_h_dim = (32, 1),
                                    seq_in = True,

        )


        self.MLP = nn.Sequential(nn.Linear(in_channels, hidden_dim), 
                        nn.BatchNorm1d(hidden_dim), 
                        nn.ELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Dropout(p=dropout_rate),
                        nn.Linear(hidden_dim,hidden_dim))
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, graph):
        
        h_g = self.gvp_model((graph.node_s,graph.node_v),graph.edge_index,(graph.edge_s,graph.edge_v),graph.seq)

        h_g= self.MLP(h_g)
        # h_l_pos, _ = to_dense_batch(h_l.pos, h_l.batch, fill_value=0)
        # h_t_pos, _ = to_dense_batch(h_t.pos, h_t.batch, fill_value=0)
        return h_g

class gvp_cl(nn.Module):
    # gvp for encode the res 
    # mate for rest param
    # mlp
    def __init__(self,config,model, dropout_rate=0.15,
                    dist_threhold=1000):
        super().__init__()
        self.config = config
        in_channels = config.n_embd
        hidden_dim = config.n_embd
        self.inter_gvp = model
        self.AA_pre = nn.Sequential(nn.Linear(in_channels, hidden_dim), 
                                                       nn.BatchNorm1d(hidden_dim), 
                        nn.ELU(),
                        nn.Linear(hidden_dim, 640),)     

        # self.type_pre = nn.Sequential(nn.Linear(in_channels, hidden_dim), 
        #                                                nn.BatchNorm1d(hidden_dim), 
        #                 nn.ELU(),
        #                 nn.Linear(hidden_dim, 3),)  
        self.dist_threhold = dist_threhold
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM,torch.nn.Embedding)
        # blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        blacklist_weight_modules = (torch.nn.LayerNorm)
        for mn, m in self.named_modules():
            if 'atbmodel' in mn:
                continue
            else:
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                    if pn.endswith('bias') or ('bias' in pn):
                        no_decay.add(fpn)
                    elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))]},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate)
        return optimizer
    def forward(self, graph):
        
        h_g = self.inter_gvp(graph)
        # h_l_x, l_mask = to_dense_batch(h_g, graph.batch, fill_value=0) # B*N*H
        # B, N_l, C_out = h_l_x.size()


        AA_type = self.AA_pre(h_g)

        i_type = None
        # h_l_pos, _ = to_dense_batch(h_l.pos, h_l.batch, fill_value=0)
        # h_t_pos, _ = to_dense_batch(h_t.pos, h_t.batch, fill_value=0)
        return h_g,AA_type,i_type


class Inter_gvp_pre(nn.Module):
    # gvp for encode the res 
    # mate for rest param
    # mlp
    def __init__(self,config,model, dropout_rate=0.15,
                    dist_threhold=1000):
        super().__init__()
        self.config = config
        in_channels = config.n_embd
        hidden_dim = config.n_embd
        self.inter_gvp = model
        self.edge_mlp = nn.Sequential(nn.Linear(in_channels, hidden_dim), 
                        nn.BatchNorm1d(hidden_dim), 
                        nn.ELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Dropout(p=dropout_rate),
                        nn.Linear(hidden_dim,hidden_dim)) 
        self.AA_pre = nn.Sequential(nn.Linear(in_channels, hidden_dim), 
                                                       nn.BatchNorm1d(hidden_dim), 
                        nn.ELU(),
                        nn.Linear(hidden_dim, 31),)     
        self.inter_pre = nn.Sequential(nn.Linear(in_channels, hidden_dim), 
                                                       nn.BatchNorm1d(hidden_dim), 
                        nn.ELU(),
                        nn.Linear(hidden_dim, 9),)    
        self.dist_pre = nn.Sequential(nn.Linear(in_channels, hidden_dim), 
                                                       nn.BatchNorm1d(hidden_dim), 
                        nn.ELU(),
                        nn.Linear(hidden_dim, 1),)  
        self.connect_pre = nn.Sequential(nn.Linear(in_channels, hidden_dim), 
                                                       nn.BatchNorm1d(hidden_dim), 
                        nn.ELU(),
                        nn.Linear(hidden_dim, 2),)
        self.type_pre = nn.Sequential(nn.Linear(in_channels, hidden_dim), 
                                                       nn.BatchNorm1d(hidden_dim), 
                        nn.ELU(),
                        nn.Linear(hidden_dim, 6),)  
        self.dist_threhold = dist_threhold
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM,torch.nn.Embedding)
        # blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        blacklist_weight_modules = (torch.nn.LayerNorm)
        for mn, m in self.named_modules():
            if 'atbmodel' in mn:
                continue
            else:
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                    if pn.endswith('bias') or ('bias' in pn):
                        no_decay.add(fpn)
                    elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))]},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate)
        return optimizer
    def forward(self, graph):
        
        h_g = self.inter_gvp(graph)
        # h_l_x, l_mask = to_dense_batch(h_g, graph.batch, fill_value=0) # B*N*H
        # B, N_l, C_out = h_l_x.size()
        # edge = h_l_x.unsqueeze(1)+h_l_x.unsqueeze(2)
        mask_ind= []
        mask_edge = []
        for mask in graph.mask_index:
            mask_ind.extend(mask)
        for mask in graph.edge_mask:
            mask_edge.extend(mask)
        h_g_n = h_g[np.array(mask_ind)==1]
        h_g_e = (h_g[graph.edge_index[0]]+h_g[graph.edge_index[1]])
        h_g_e = h_g_e[np.array(mask_edge)==1]
        AA_type = self.AA_pre(h_g_n)
        inter = self.inter_pre(h_g_n)
        dist = self.dist_pre(h_g_e)
        connect = self.connect_pre(h_g_e)
        i_type = self.type_pre(h_g_e)
        # h_l_pos, _ = to_dense_batch(h_l.pos, h_l.batch, fill_value=0)
        # h_t_pos, _ = to_dense_batch(h_t.pos, h_t.batch, fill_value=0)
        return AA_type,inter,dist,connect,i_type

    def compute_euclidean_distances_matrix(self, X, Y):
        X = X.float()
        Y = Y.float()
        return     torch.sqrt(((X.unsqueeze(-2)-Y.unsqueeze(-3))**2).sum(-1))
    def discount_num(self,C_batch,B):
        c_dis = {}
        for i in C_batch:
            d= int(i) #typebug
            if d in c_dis:
                c_dis[d]+=1
            else:
                c_dis[d]=1
        # pdb.set_trace()
        return torch.tensor(list(c_dis.values()),dtype=torch.float,requires_grad=False).reshape(B,1)
    




class pairdatt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.Sequential(*[ Multhead_attention(config)for _ in range(config.n_layer)])
    def forward(self, x,y,mask=None):
        attn_maps=[]
        for layer in self.blocks:
            x1, attn = layer(x,y)
            
            x2, attn = layer(y,x)  
            x = x2
            y = x1

        # out = torch.concat([x1[:,0,:],x2[:,0,:]],dim=-1)
        return x1,x2
class predictor(nn.Module):
    def __init__(self, config,atbmodel):
        super().__init__()

        self.pariatt = pairdatt(config)
        self.esm_transfom = nn.Linear(640,config.n_embd)
        self.proj = nn.Sequential(nn.Linear(config.n_embd*2, config.n_embd*4), 
                nn.GELU(),
                nn.Linear(config.n_embd*4, config.n_embd*2),
                nn.Dropout(p=config.embd_pdrop))
        self.ffn = nn.Linear(config.n_embd*2,1)

        self.atbmodel = atbmodel
        for param in self.atbmodel.parameters():
            param.requires_grad = False
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM,torch.nn.Embedding)
        # blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        blacklist_weight_modules = (torch.nn.LayerNorm)
        for mn, m in self.named_modules():
            if 'atbmodel' in mn:
                continue
            else:
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                    if pn.endswith('bias') or ('bias' in pn):
                        no_decay.add(fpn)
                    elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))]},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate)
        return optimizer
    def forward(self,atg_embd,atg_m_embd,atb_h,atb_l ):
        atb_embd,_,_,_ = self.atbmodel(atb_h,atb_l)
        atb_m_embd = atb_embd.clone()
        attn_maps=[]
        atg_embd = self.esm_transfom(atg_embd)
        atg_m_embd = self.esm_transfom(atg_m_embd)
        # atg_m_embd = atg_m_embd
        # out = self.pariatt(atg_embd,atb_embd)
        # out_m = self.pariatt(atg_m_embd,atb_m_embd) 
        # if use ddp all out must output in return 
        out1,out2 = self.pariatt(atg_embd,atb_embd)
        out_m1,out_m2 = self.pariatt(atg_m_embd,atb_m_embd)
        out = torch.concat([out1[:,0,:],out2[:,0,:]],dim=-1)
        out_m = torch.concat([out_m1[:,0,:],out_m2[:,0,:]],dim=-1)
        out = self.proj(out) + out
        out_m =self.proj(out_m) + out_m

        final_out = self.ffn(out-out_m)
        mask_out = None
        nsp_out = None
        return final_out,out1,out2,out_m1,out_m2
class pairatt_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.ln1 = nn.LayerNorm(config.n_embd)
        # self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = Multhead_attention(config)
        # self.mlp = nn.Sequential(
        #     nn.Linear(config.n_embd, 4 * config.n_embd),
        #     nn.GELU(),
        #     nn.Linear(4 * config.n_embd, config.n_embd),
        #     nn.Dropout(config.resid_pdrop),
        # )
    def forward(self, x,y,mask=None):
        attnls =[]
        x1, attn = self.attn(x,y,mask)
        attnls.append(attn)
        x2, attn = self.attn(y,x,mask)
        attnls.append(attn)
        
        x = x1
        y = x2
        return x,y,attnls
class crossatt_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = Multhead_attention(config)
        # self.mlp = nn.Sequential(
        #     nn.Linear(config.n_embd, 4 * config.n_embd),
        #     nn.GELU(),
        #     nn.Linear(4 * config.n_embd, config.n_embd),
        #     nn.Dropout(config.resid_pdrop),
        # )
    def forward(self, x,y,mask=None):
        x1, attn = self.attn(self.ln1(x),self.ln1(y),mask)
        # x2, attn = self.attn(self.ln1(y),self.ln1(x),mask)
        x = x1 + self.mlp(self.ln2(x1))
        # y = x2 + self.mlp(self.ln2(x2))
        return x
class ppipredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.graph_model = graph_model
        # self.atb_model = atb_model
        self.pariatt = pairatt_block(config)
        self.crossatt = Multhead_attention(config)
        self.esm_transfom = nn.Linear(640,config.n_embd)
        self.proj = nn.Sequential(nn.Linear(config.n_embd, config.n_embd), 
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd),
                nn.Dropout(p=config.embd_pdrop))
        self.mlp = nn.Sequential(nn.Linear(config.n_embd*2, config.n_embd*4), 
                nn.GELU(),
                nn.Linear(config.n_embd*4, config.n_embd*4),
                nn.Dropout(p=config.embd_pdrop),
                nn.Linear(config.n_embd*4, config.n_embd*2))
        self.ffn = nn.Linear(config.n_embd*2,1)

        # self.atbmodel = atbmodel
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM,torch.nn.Embedding)
        # blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        blacklist_weight_modules = (torch.nn.LayerNorm)
        for mn, m in self.named_modules():
            if 'atbmodel' in mn:
                continue
            else:
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                    if pn.endswith('bias') or ('bias' in pn):
                        no_decay.add(fpn)
                    elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))]},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate)
        return optimizer
    def forward(self,wtseq,mtseq,graphemb,mask ):

        wtseq = self.esm_transfom(wtseq)
        mtseq = self.esm_transfom(mtseq)
        # atg_m_embd = atg_m_embd
        # out = self.pariatt(atg_embd,atb_embd)
        # out_m = self.pariatt(atg_m_embd,atb_m_embd) 
        # if use ddp all out must output in return 
        wtseq,mtseq,attnls = self.pariatt(wtseq,mtseq,mask)
        # wtseq = self.crossatt(wtseq,graphemb,mask)
        # mtseq = self.crossatt(mtseq,graphemb,mask)
        wtseq = self.proj(wtseq) + wtseq
        mtseq =self.proj(mtseq) + mtseq
        final = torch.concat([mtseq-wtseq,graphemb],dim=-1)
        final = final + self.mlp(final)
        final_out = self.ffn(final)
        mask_out = None
        nsp_out = None
        return final_out,wtseq,mtseq,attnls