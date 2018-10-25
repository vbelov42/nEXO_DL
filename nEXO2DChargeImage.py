#!/usr/bin/env python
# coding: utf-8

# In[7]:


import ROOT
import numpy as np


# In[8]:


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imageio


# In[11]:


def read3dimage():
    rootfile = ROOT.TFile('electron.root','read')
    ttree = rootfile.Get('SSigTree')
    for i in range(ttree.GetEntries()):
        image_2dcharge = np.zeros((500,500,3),dtype=np.uint8)
        ttree.GetEntry(i)
        xq = ttree.xq
        xpad = ttree.fxposition
        xtile = ttree.fxqtile
        xrisetime = ttree.fxrisetime
        for j in range(len(xq)):
            H=int(xpad[j]+750)/3
            W=int(xtile[j]+750)/3
            pixelq = int(xq[j]/10/34)
            for w_iter in range(10):
                w_local = W - 5 + w_iter
                #print H, w_local, pixelq
                image_2dcharge[H,w_local,0]=pixelq + image_2dcharge[H,w_local,0]
                if xrisetime[j]>40:
                    continue
                else:
                    image_2dcharge[H,w_local,1]= xrisetime[j] + image_2dcharge[H,w_local,1]
        yq = ttree.yq
        ypad = ttree.fyposition
        ytile = ttree.fyqtile
        yrisetime = ttree.fyrisetime
        for j in range(len(yq)):
            W=int(ypad[j]+750)/3
            H=int(ytile[j]+750)/3
            pixelq = int(yq[j]/10/34)
            for h_iter in range(10):
                h_local = H - 5 + h_iter
                #print h_local, W, pixelq
                image_2dcharge[h_local,W,0]=pixelq + image_2dcharge[h_local,W,0]
                if yrisetime[j]>40:
                    continue
                else:
                    image_2dcharge[H,w_local,1]= yrisetime[j] + image_2dcharge[H,w_local,1]
        #imagearray.append(image_2dcharge)
        
        imageio.imwrite('./images/electron%d.jpg' % i, image_2dcharge)

    
    

        


# In[12]:


read3dimage()

