from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_history(history,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot([x['epoch'] for x in history],[x['train_loss'] for x in history],label='Train loss'); ax.plot([x['epoch'] for x in history],[x['val_loss'] for x in history],label='Validation loss'); ax.set(xlabel='Epoch',ylabel='BCE + Dice loss'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def plot_effort(summary,path,metric):
    Path(path).parent.mkdir(parents=True,exist_ok=True); x=sorted(map(int,summary.keys())); y=[summary[str(i)][metric] for i in x]
    fig,ax=plt.subplots(figsize=(6,4)); ax.plot(x,y,marker='o',linewidth=2,color='#087f8c'); ax.set(xlabel='Simulated click budget',ylabel=metric.upper(),ylim=(0,1)); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


@torch.no_grad()
def save_prediction_grid(model,dataset,ids,device,path):
    if not ids: return
    model.eval(); count=min(4,len(ids)); fig,axs=plt.subplots(count,3,figsize=(9,3*count)); axs=np.atleast_2d(axs)
    for r,idx in enumerate(ids[:count]):
        image,mask,_=dataset[idx]; x=image[None].to(device); zeros=torch.zeros_like(x); pred=torch.sigmoid(model(torch.cat([x,zeros,zeros,zeros],1)))[0,0].cpu().numpy()
        axs[r,0].imshow(image[0],cmap='gray'); axs[r,0].set_title('X-ray'); axs[r,1].imshow(mask[0],cmap='gray'); axs[r,1].set_title('Ground truth'); axs[r,2].imshow(pred>=.5,cmap='gray'); axs[r,2].set_title('Initial prediction')
        for ax in axs[r]: ax.axis('off')
    fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=150); plt.close(fig)
