import argparse, csv, random, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from models.unet import UNet
from models.losses import bce_dice_loss, dice_score
from dataset.dataloader import discover_pairs, LungDataset
from interaction.click_generator import sample_error_click, add_click
from utils.helpers import seed_all, save_json, get_device
from utils.visualization import save_prediction_grid


def split_ids(n, seed):
    ids = list(range(n)); random.Random(seed).shuffle(ids)
    a = max(1, int(.70*n)); b = max(a+1, int(.85*n))
    return ids[:a], ids[a:b], ids[b:]


def epoch(model, loader, device, optimizer=None, steps=2):
    train = optimizer is not None; model.train(train); losses=[]; dices=[]
    for image, target, _ in loader:
        image=image.to(device); target=target.to(device)
        pos=torch.zeros_like(target); neg=torch.zeros_like(target); prev=torch.zeros_like(target)
        if train: optimizer.zero_grad(set_to_none=True)
        total=0
        # Ground-truth-derived clicks are simulated annotation, only during train/validation.
        for k in range(steps):
            logits=model(torch.cat([image,pos,neg,prev],1))
            loss=bce_dice_loss(logits,target)
            total=total+loss/(k+1)
            probs=torch.sigmoid(logits)
            if k+1 < steps:
                for i in range(len(image)):
                    c=sample_error_click((probs[i:i+1]>.5).float(),target[i:i+1])
                    pos[i:i+1],neg[i:i+1]=add_click(pos[i:i+1],neg[i:i+1],c)
                prev=probs.detach()
        if train: total.backward(); optimizer.step()
        losses.append(float(total.detach())); dices.extend(dice_score(logits.detach(),target).cpu().tolist())
    return sum(losses)/max(len(losses),1), sum(dices)/max(len(dices),1)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--size',type=int,default=256); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--seed',type=int,default=42); p.add_argument('--patience',type=int,default=5); p.add_argument('--steps',type=int,default=2); a=p.parse_args()
    seed_all(a.seed); device=get_device()
    pairs=discover_pairs(a.data)
    if len(pairs)<10: raise RuntimeError(f'Only {len(pairs)} matched pairs; need at least 10 for train/validation/test split.')
    tr,va,te=split_ids(len(pairs),a.seed); ds=LungDataset(pairs,a.size)
    loader=lambda ids,shuffle: DataLoader(Subset(ds,ids),batch_size=a.batch_size,shuffle=shuffle,num_workers=0,pin_memory=device.type=='cuda')
    train_loader=loader(tr,True); val_loader=loader(va,False)
    model=UNet(base=24).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=2,factor=.5)
    Path('checkpoints').mkdir(exist_ok=True); Path('results/graphs').mkdir(parents=True,exist_ok=True); Path('results/predictions').mkdir(parents=True,exist_ok=True)
    config=vars(a)|{'device':str(device),'pairs':len(pairs),'split_counts':{'train':len(tr),'validation':len(va),'test':len(te)},'input_channels':['xray','positive_click_map','negative_click_map','previous_prediction'],'base_channels':24}
    save_json('results/config.json',config); history=[]; best=-1; stale=0; start=time.time()
    for e in range(1,a.epochs+1):
        tl,td=epoch(model,train_loader,device,opt,a.steps); vl,vd=epoch(model,val_loader,device,None,a.steps); sched.step(vd)
        row={'epoch':e,'train_loss':tl,'train_dice':td,'val_loss':vl,'val_dice':vd,'lr':opt.param_groups[0]['lr']}; history.append(row); print(row)
        torch.save({'model':model.state_dict(),'base':24,'config':config,'epoch':e,'val_dice':vd},'checkpoints/latest.pt')
        if vd>best:
            best=vd; stale=0; torch.save({'model':model.state_dict(),'base':24,'config':config,'epoch':e,'val_dice':vd},'checkpoints/best.pt')
            save_prediction_grid(model,ds,va,device,'results/predictions/validation.png')
        else: stale+=1
        save_json('results/training_history.json',history)
        with open('results/training_history.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=history[0]); w.writeheader(); w.writerows(history)
        from utils.visualization import plot_history
        plot_history(history,'results/graphs/training_history.png')
        if stale>=a.patience: print(f'Early stopping at epoch {e}.'); break
    print(f'Training complete in {time.time()-start:.1f}s; best validation Dice={best:.4f}; test split held out ({len(te)} images).')

if __name__=='__main__': main()
