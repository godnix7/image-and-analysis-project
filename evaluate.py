import argparse, json, time, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataset.dataloader import discover_pairs, LungDataset
from interaction.click_generator import sample_error_click, add_click
from utils.helpers import load_checkpoint, save_json, get_device, synchronize_device
from utils.metrics import binary_metrics
from utils.visualization import plot_effort


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--checkpoint',default='checkpoints/best.pt'); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    random.seed(a.seed)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); device=get_device()
    pairs=discover_pairs(a.data); n=len(pairs); ids=list(range(n)); random.Random(a.seed).shuffle(ids); test_ids=ids[max(1,int(.85*n)):]
    if not test_ids: raise RuntimeError('No held-out test samples; dataset too small.')
    ds=LungDataset(pairs,ck['config']['size']); model,_=load_checkpoint(a.checkpoint,device); loader=DataLoader(Subset(ds,test_ids),batch_size=1,shuffle=False)
    budgets=[0,1,3,5,10]; results={b:[] for b in budgets}; per_image=[]; inference=[]
    for image,target,path in loader:
        image=image.to(device); target=target.to(device); pos=torch.zeros_like(target); neg=torch.zeros_like(target); prev=torch.zeros_like(target); row={'image':path[0]}; previous=0
        for budget in budgets:
            for _ in range(budget-previous):
                logits=model(torch.cat([image,pos,neg,prev],1)); pred=(torch.sigmoid(logits)>.5).float(); c=sample_error_click(pred,target); pos,neg=add_click(pos,neg,c); prev=torch.sigmoid(logits)
            previous=budget
            synchronize_device(device)
            t=time.perf_counter(); logits=model(torch.cat([image,pos,neg,prev],1))
            synchronize_device(device)
            inference.append((time.perf_counter()-t)*1000); prev=torch.sigmoid(logits)
            m=binary_metrics((prev[0,0].cpu().numpy()>=.5),target[0,0].cpu().numpy()); results[budget].append(m); row[str(budget)]=m
        per_image.append(row)
    summary={str(b):{k:float(np.mean([x[k] for x in vals])) for k in vals[0]} for b,vals in results.items()}
    out={'dataset_pairs':n,'test_count':len(test_ids),'test_indices':test_ids,'checkpoint':a.checkpoint,'device':str(device),'mean_inference_ms':float(np.mean(inference)),'interaction_budget':summary,'per_image':per_image,'protocol':'Simulated oracle clicks sampled from largest false-negative/false-positive region using test ground truth; used only for offline effort curve.'}
    Path('results/evaluation').mkdir(parents=True,exist_ok=True); save_json('results/evaluation/metrics.json',out); plot_effort(summary,'results/graphs/effort_vs_dice.png','dice'); plot_effort(summary,'results/graphs/effort_vs_iou.png','iou'); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
