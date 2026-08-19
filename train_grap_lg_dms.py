
import math
import logging
import os
import scipy
from tqdm import tqdm
import numpy as np
from model.models import myconfig, ppipredictor
from dataset.predictorset import dms_dataset,ppi_collate_fn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
import re
import pandas as pd
# from rdkit import Chem
# from utils import sample
from torch.nn import functional as F
import logging
import argparse
import random
import pickle
from torch import nn
import torch.multiprocessing as mp
# import esm
# from esm.pretrained import load_model_and_alphabet_core
import json
import pdb
# for ddp
from ddpfc import init_ddp,get_ddp_generator
import torch.distributed as dist
logger = logging.getLogger(__name__)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-3
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    use_scaler = False
    num_workers = 0 # for DataLoader

    def __init__(self, configdic={},**kwargs):
        for k,v in configdic.items():
            setattr(self, k, v)
        for k,v in kwargs.items():
            setattr(self, k, v)
class Trainer:

    def __init__(self, model, train_dataset, val_dataset,test_dataset, config):
        self.device = config.device
        # self.model = model.to(self.device)
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.val_dataset = val_dataset
        self.use_scaler = config.use_scaler

        num_gpus=4
        if torch.cuda.is_available():
            #self.devices = [try_gpu(i) for i in range(num_gpus)]

            #self.model = torch.nn.DataParallel(self.model,device_ids=self.devices).to(self.device)
            self.device = torch.cuda.current_device()
            # self.model = self.model.to(self.device)
    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), './ckpt/'+self.config.ckpt_path+'.pt')

    def train(self):
        model, config = self.model, self.config
        #for ddp
        scaler = GradScaler()
        if config.parallel:
            local_rank = int(os.environ['LOCAL_RANK'])

            init_ddp(local_rank)

            model = model.cuda()
            optimizer = model.configure_optimizers(config)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)
        else:
            torch.cuda.set_device(config.device)
            model = model.cuda()
            optimizer = model.configure_optimizers(config)
        if config.lr_decay:
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[range(0,80,2)], gamma=0.90)
                # 1e-3 to 1e-5


        def run_epoch(split,epoch,use_scaler=False):
            is_train = split == 'train'
            is_val = split =='val'
            model.train(is_train)
            if is_train:
                dataset=self.train_dataset
            elif is_val:
                dataset=self.val_dataset
            else:
                dataset=self.test_dataset
            #for ddp
            if config.parallel:
                local_rank = int(os.environ['LOCAL_RANK'])
                g = get_ddp_generator()
                sampler = torch.utils.data.distributed.DistributedSampler(dataset) 
                loader = DataLoader(dataset, shuffle=False, pin_memory=True,collate_fn = ppi_collate_fn,
                            batch_size=config.batch_size,num_workers= config.num_workers,sampler=sampler,generator=g)
                loader.sampler.set_epoch(epoch)
            else:
                loader = DataLoader(dataset, shuffle=True, pin_memory=True,collate_fn = ppi_collate_fn,
                            batch_size=config.batch_size,num_workers= config.num_workers)

            losses = []

            # pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            if not config.parallel or local_rank ==0:
                pbar =tqdm(enumerate(loader), total=len(loader)) 
            else:
                pbar =enumerate(loader)
            # pbar = tqdm(enumerate(loader), total=len(loader))
            average_smi=[]
            same=[]
            losses_a=[]
            losses_b=[]
            losses_c=[]
            # torch.autograd.set_detect_anomaly(True)
            for it, (wtseq,mtseq,gp,label,mask,valen) in pbar:

                device = next(model.parameters()).device
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        # g = g.cuda()
                        m= nn.Sigmoid()
                        out,_,_= model(wtseq.to(torch.float32).cuda(),mtseq.to(torch.float32).cuda(),gp.to(torch.float32).cuda(),mask.cuda())

                        out = m(out.mean(1))
                        def loss_fn(out,label):


                            return F.mse_loss(out,label.unsqueeze(1).to(out.dtype).cuda())
                        
                        loss  = loss_fn(out,label)
                        loss = loss.mean()

                        if is_train:

                            model.zero_grad()
                            if use_scaler:
                                # loss.backward()
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                        if config.parallel:
                            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
                        losses.append(loss.item())
            if is_train and config.lr_decay:

                        scheduler.step()

            return np.mean(losses)
            # return np.mean(losses)
                    
                    

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        molecules = []
        train_losses=[]
        test_losses = []
        val_losses = []
        
        for epoch in range(config.max_epochs):
            val_loss = run_epoch('val',epoch,use_scaler=self.use_scaler)
            train_loss = run_epoch('train',epoch,use_scaler=self.use_scaler)

            test_loss = run_epoch('test',epoch,use_scaler=self.use_scaler)
                # wandb.log({'epoch_valid_loss': test_loss, 'epoch_train_loss': train_loss,'epoch_valid_loss_a': test_loss_a, 'epoch_train_loss_a': train_loss_a,'epoch_valid_loss_b': test_loss_b, 'epoch_train_loss_b': train_loss_b, 'epoch': epoch + 1})
            # supports early stopping based on the test loss, or just save always if no test set is provided
            if not config.parallel or local_rank ==0:
                good_model = self.test_dataset is None or val_loss < best_loss
                if self.config.ckpt_path is not None and good_model:
                    best_loss = val_loss
                    print(f'Saving at epoch {epoch + 1}')
                    self.save_checkpoint()
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                val_losses.append(val_loss)
                result = pd.DataFrame([train_losses,test_losses,val_losses]).T.to_csv('./result/'+self.config.ckpt_path+'.csv')
        if config.parallel:
            dist.destroy_process_group()
        return result

if __name__=='__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_path', type=str,
                         required=False)
    parser.add_argument('-p','--pdbcheck_path', type=str,default='./dmscheckclean.csv',
                        help="pdbcheck path", required=False)
    parser.add_argument('-c','--config_path', type=str,
                        help="config path", required=False)
    parser.add_argument('-s','--seq_emb_path', type=str,
                        help="seq_emb path", required=False)
    parser.add_argument('-seed', type=int,default=42,
                        help="seed", required=False)
    args = parser.parse_args()
    set_seed(args.seed)
    with open(args.config_path,'r') as f:
        strf = f.read()
        config_log = json.loads(strf)

    
    data = pd.read_csv(config_log['data_config']['data_path'])
    data2 = pd.read_csv(args.pdbcheck_path)
    antb2 = data2.antibody.to_list()
    Rchain = data2.Rchain.to_list()
    Lchain = data2.lchain.to_list()
    Hchain = data2.Hchain.to_list()
    antbdic ={}
    for i in range(len(antb2)):
        antbdic[antb2[i]] = f'{crys[i]}_{Rchain[i]}_{Hchain[i]}{Lchain[i]}'
    with open(config_log['data_config']['model_emb_path'],'rb') as f:
        graph_emb = pickle.load(f) 
    with open(config_log['data_config']['emb_path']+'wtseq_emb.bin','rb') as f:
        wtseq_emb = pickle.load(f) 
    with open(config_log['data_config']['seq_emb_path'],'rb')as f:
        esm_emb = pickle.load(f)
    set_seed(64)
    index = []
    index2 = []
    for i in range(len(data)):
        if np.array(data.iloc[i,1]) =='REGN10987' or np.array(data.iloc[i,1]) == 'LY-CoV016':
            index2.append(i)
        else:
            index.append(i)
    random.shuffle(index)
    trainindex = index[:int(0.8*len(index))]
    # testindex = index[int(0.8*len(index)):int(0.9*len(index))]
    testindex = index[int(0.8*len(index)):int(0.9*len(index))]
    valindex = index[int(0.9*len(index)):]
    def filter_seqemb(data,seq_emb):
        return np.array(seq_emb)[np.array(data.index),:]
    def fil(index,seq_emb):
        return np.array(seq_emb)[index,:]   
    train_dataset = dms_dataset(config_log['data_config'],data.iloc[trainindex,:].reset_index(drop=True),esm_emb,graph_emb,wtseq_emb,data2,antbdic) 
    test_dataset = dms_dataset(config_log['data_config'],data.iloc[testindex,:].reset_index(drop=True),esm_emb,graph_emb,wtseq_emb,data2,antbdic)     
    val_dataset = dms_dataset(config_log['data_config'],data.iloc[valindex,:].reset_index(drop=True),esm_emb,graph_emb,wtseq_emb,data2,antbdic) 
    model_config = myconfig(configdic=config_log['model_config'])

    model = ppipredictor(model_config)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    if model_config.is_continue:
        model.load_state_dict(torch.load(model_config.continue_path+'.pt',map_location='cpu'))
        print('continue')
    trainer_config = TrainerConfig(configdic = config_log['trainer_config'])
    trainer  = Trainer(model=model,config = trainer_config,train_dataset=train_dataset,test_dataset=test_dataset,val_dataset=val_dataset)
    result = trainer.train()
    with open(f'./ckpt/{trainer_config.ckpt_path}_config.jason','w') as f:
        json.dump(config_log,f)
