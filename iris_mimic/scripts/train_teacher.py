"""Train IRIS with offline Qwen utilities, preserving original IRIS source files."""
import argparse, json, os
from pathlib import Path
import numpy as np, torch, lancedb
import torch.nn.functional as F
from transformers import AutoConfig
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from iris.config import get_config
from iris.dataset import LanceTableDataset, get_data_splits, get_ini_queries, get_num_classes
from iris.model import make_model
from iris.loss import QueryLoss

class ScoredDataset(LanceTableDataset):
    def __init__(self, scores, **kwargs): super().__init__(**kwargs); self.scores=scores
    def __getitem__(self,idx):
        doc_id=self.ids[idx]; df=self.search_chunks(doc_id)
        if df.height==0:
            return torch.zeros(1,self.vector_dim),torch.tensor(self.ignore_index),torch.zeros(1),torch.zeros(1,dtype=torch.bool),torch.full((1,),-1,dtype=torch.long)
        vectors=df['vector'].to_torch().float(); label=torch.tensor(int(df['label'][0]))
        utilities=[]; valid=[]
        for cid in df['chunk_id'].to_list():
            key=(str(doc_id),str(cid)); utilities.append(self.scores.get(key,0.)); valid.append(key in self.scores)
        return vectors,label,torch.tensor(utilities,dtype=torch.float32),torch.tensor(valid,dtype=torch.bool),torch.tensor(df["query_id"].to_list(),dtype=torch.long)

def collate(batch):
    v,y,u,m,q=zip(*batch)
    v=pad_sequence(v,batch_first=True,padding_value=0)
    u=pad_sequence(u,batch_first=True,padding_value=0)
    m=pad_sequence(m,batch_first=True,padding_value=False)
    lengths=torch.tensor([len(z[0]) for z in batch]); padmask=torch.arange(v.size(1))[None,:]<lengths[:,None]
    q=pad_sequence(q,batch_first=True,padding_value=-1)
    return v,torch.stack(y),padmask,u,m,q

def pairwise_loss(vectors,queries,utilities,valid,padding,query_ids,delta=.05,tau=1.):
    # Per-query differentiable IRIS retrieval scores r=-||x-q||^2 / d.
    r=-(vectors.unsqueeze(1)-queries[None,:,None,:]).square().sum(-1)/vectors.shape[-1]
    du=utilities[:,:,None]-utilities[:,None,:]
    pair=(du>delta)&valid[:,:,None]&valid[:,None,:]&padding[:,:,None]&padding[:,None,:]&(query_ids[:,:,None]==query_ids[:,None,:])
    if not pair.any(): return r.sum()*0
    dr=r[:,:,:,None]-r[:,:,None,:]
    losses=F.softplus(-dr/tau)
    return losses.masked_select(pair[:,None,:,:].expand_as(losses)).mean()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lambda-teacher',type=float,required=True);ap.add_argument('--scores',default='data/mimic/teacher_scores.jsonl');ap.add_argument('--config',default='config.mimic.toml');ap.add_argument('--output',required=True);args=ap.parse_args()
    if args.lambda_teacher<=0: raise ValueError('This script is for positive teacher lambda')
    scores={}
    for line in open(args.scores):
        row=json.loads(line);scores[(str(row['id']),str(row['chunk_id']))]=float(row['utility'])
    if not scores: raise RuntimeError('No teacher scores found')
    cfg=get_config(args.config)
    torch.manual_seed(cfg.seed);np.random.seed(cfg.seed)
    cfg.model.d_model=AutoConfig.from_pretrained(cfg.model_id).hidden_size
    tbl=lancedb.connect(str(cfg.database.path)).open_table(cfg.dataset.table_name)
    cfg.model.output_dim=get_num_classes(tbl)
    train_ids,val_ids,test_ids=get_data_splits(tbl,cfg.seed)
    queries=get_ini_queries(tbl,cfg)
    def loader(ids,shuffle):
        ds=ScoredDataset(scores=scores,ids=ids,tbl=tbl,vector_dim=cfg.model.d_model,k_neighbors=cfg.database.k_neighbors,ignore_index=cfg.trainer.ignore_index,ini_queries=queries.copy())
        return DataLoader(ds,batch_size=cfg.dataloader.batch_size,shuffle=shuffle,num_workers=0,collate_fn=collate)
    train=loader(train_ids,True);val=loader(val_ids,False)
    model=make_model(queries,cfg.model)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    optimizer=AdamW(model.parameters(),lr=cfg.optimizer.lr,weight_decay=cfg.optimizer.weight_decay)
    criterion=QueryLoss(cfg.trainer.temperature,cfg.trainer.ignore_index)
    for epoch in range(cfg.trainer.max_epochs):
        model.train();running=0;count=0
        for v,y,mask,u,valid,qids in train:
            v,y,mask,u,valid,qids=[t.to(device) for t in (v,y,mask,u,valid,qids)]
            optimizer.zero_grad()
            logits=model(v,mask)
            base=criterion(logits,y,model.queries)
            teacher=pairwise_loss(v,model.queries,u,valid,mask,qids)
            loss=base+args.lambda_teacher*teacher
            loss.backward();clip_grad_norm_(model.parameters(),cfg.trainer.gradient_clip);optimizer.step()
            train.dataset.queries=model.queries.detach().cpu().numpy();running+=float(loss.detach());count+=1
        model.eval();correct=total=0
        val.dataset.queries=model.queries.detach().cpu().numpy()
        with torch.no_grad():
            for v,y,mask,u,valid,qids in val:
                v,y,mask=[t.to(device) for t in (v,y,mask)]
                pred=model(v,mask).argmax(1)
                good=y!=cfg.trainer.ignore_index
                correct+=int(((pred==y)&good).sum());total+=int(good.sum())
        print(f'epoch={epoch+1} lambda={args.lambda_teacher} train_loss={running/max(count,1):.5f} val_accuracy={correct/max(total,1):.4f}',flush=True)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),out)
    print('SAVED',out,flush=True)
if __name__=='__main__': main()
