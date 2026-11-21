import argparse
import numpy as np
from PIL import Image
import torch
from utils.helpers import load_checkpoint, get_device


def main():
    p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--checkpoint',default='checkpoints/best.pt'); p.add_argument('--output',default='prediction.png'); a=p.parse_args()
    device=get_device(); model,ck=load_checkpoint(a.checkpoint,device); size=ck['config']['size']; im=Image.open(a.image).convert('L'); original=im.size; im=im.resize((size,size),Image.Resampling.BILINEAR); x=torch.from_numpy(np.asarray(im,dtype=np.float32)[None,None]/255).to(device); zero=torch.zeros_like(x)
    with torch.no_grad(): prob=torch.sigmoid(model(torch.cat([x,zero,zero,zero],1)))[0,0].cpu().numpy()
    mask=Image.fromarray((prob>=.5).astype('uint8')*255).resize(original,Image.Resampling.NEAREST); mask.save(a.output); print(f'Saved binary mask to {a.output}')

if __name__=='__main__': main()
