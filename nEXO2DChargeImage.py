#!/usr/bin/env python
import ROOT
import os
import numpy as np
import imageio

def read3dimage():
    ttree = ROOT.TChain('SSigTree')
    ttree.Add('/gpfs/loomis/scratch.omega/hep/david_moore/zl423/tree/Aug18/3mm_10cm/electron_2458_run*root')
    maxlen = 0 #1540
    maxq = 0   #91228
    filenum = 0
    for i in range(ttree.GetEntries()):
        image_2dcharge = np.zeros((200,255,3),dtype=np.uint8)
        ttree.GetEntry(i)
        fxwf = ttree.fxwf
        fywf = ttree.fywf
        xposition = ttree.fxposition
        xpos = []
        for j in range(len(xposition)):
            xpos.append(xposition[j])
        yposition = ttree.fyposition
        ypos = []
        for j in range(len(yposition)):
            ypos.append(yposition[j])
        xmin = 0
        xmax = 0
        if len(xpos)>0:
            xmin = min(xpos)
            xmax = max(xpos)
        ymin = 0
        ymax = 0
        if len(ypos)>0:
            ymin = min(ypos)
            ymax = max(ypos)
        if ymax - ymin > 500 or xmax - xmin > 500:
            continue
        filenum +=1
        for m in range(len(xposition)):
            H = int((xposition[m] - xmin)/3)+10
            for n in range(len(fxwf[m])/6):
                if n > 254:
                    break
                image_2dcharge[H, n, 0] += (fxwf[m][6*n]/40. + 25)
        for m in range(len(yposition)):
            H = int((yposition[m] - ymin)/3)+10
            for n in range(len(fywf[m])/6):
                if n > 254:
                    break
                image_2dcharge[H, n, 1] += (fywf[m][6*n]/40. + 25)

        imageio.imwrite('./images/electron%d.jpg' % filenum, image_2dcharge)
    print maxlen, maxq

read3dimage()

