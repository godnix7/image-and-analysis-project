import tempfile, unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models.unet import UNet
from models.losses import bce_dice_loss
from dataset.dataloader import discover_pairs, LungDataset
from interaction.click_generator import add_click
from utils.metrics import binary_metrics
from utils.helpers import load_checkpoint


class CoreTests(unittest.TestCase):
    def test_pair_discovery_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'CXR_png').mkdir(); (root/'masks').mkdir()
            Image.fromarray(np.full((30,40),120,np.uint8)).save(root/'CXR_png'/'case_01.png')
            m=np.zeros((30,40),np.uint8); m[4:20,5:30]=255; Image.fromarray(m).save(root/'masks'/'case_01_mask.png')
            pairs=discover_pairs(root); self.assertEqual(len(pairs),1)
            x,y,_=LungDataset(pairs,64)[0]; self.assertEqual(tuple(x.shape),(1,64,64)); self.assertEqual(tuple(y.shape),(1,64,64)); self.assertTrue(set(torch.unique(y).tolist())<={0.,1.})

    def test_network_loss(self):
        model=UNet(base=4); x=torch.randn(2,4,64,64); out=model(x); self.assertEqual(tuple(out.shape),(2,1,64,64)); loss=bce_dice_loss(out,torch.zeros_like(out)); self.assertTrue(torch.isfinite(loss))

    def test_metrics_perfect(self):
        a=np.array([[0,1],[1,0]]); self.assertAlmostEqual(binary_metrics(a,a)['dice'],1.0)

    def test_click_encoding(self):
        a=torch.zeros(1,1,32,32); b=torch.zeros_like(a); a,b=add_click(a,b,(12,17,True)); self.assertGreater(float(a[0,0,17,12]),0.99); self.assertEqual(float(b.sum()),0)

    def test_checkpoint_roundtrip_and_prediction_refinement(self):
        with tempfile.TemporaryDirectory() as td:
            model=UNet(base=4)
            path=Path(td)/'tiny.pt'
            torch.save({'model':model.state_dict(),'base':4,'config':{'size':32}},path)
            loaded,_=load_checkpoint(path,torch.device('cpu'))
            image=torch.rand(1,1,32,32); pos=torch.zeros_like(image); neg=torch.zeros_like(image); prior=torch.zeros_like(image)
            with torch.no_grad():
                first=torch.sigmoid(loaded(torch.cat([image,pos,neg,prior],1)))
                pos,neg=add_click(pos,neg,(16,16,True)); refined=torch.sigmoid(loaded(torch.cat([image,pos,neg,first],1)))
            self.assertEqual(tuple(refined.shape),(1,1,32,32)); self.assertTrue(torch.isfinite(refined).all())

if __name__=='__main__': unittest.main()
