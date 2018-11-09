#!/usr/bin/env python
import ROOT
import os
import numpy as np
from scipy.misc import imsave

def read3dimage(pitch, simtype):
    ttree = ROOT.TChain('SSigTree')
    if simtype == 'signal':
        ttree.Add('/p/lscratchh/nexouser/zpli/tree_new/%dmm/bb0n/*root' % pitch)
    else:
        ttree.Add('/p/lscratchh/nexouser/zpli/tree_new/%dmm/*root' % pitch)

    maxlen = 0 #1540
    maxq = 0   #91228
    filenum = 0
    iteration = 0
    for i in range(ttree.GetEntries()):
        iteration += 1
        if filenum > 60000:
            break
        image_2dcharge = np.zeros((200,255,3),dtype=np.uint8)
        ttree.GetEntry(i)
        if np.sqrt(ttree.avedx**2+ttree.avedy**2) > 25:
            continue
        escale = 91100./ttree.charge
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
        if ymax - ymin > pitch*180 or xmax - xmin > pitch*180:
            continue
        filenum +=1
        for m in range(len(xposition)):
            H = int((xposition[m] - xmin)/pitch)+10
            wflen = len(fxwf[m])
            samplet = 0
            for n in range(255):
                if n < 80:
                    samplet = 2*n
                elif n < 150:
                    samplet = 160 + 4*(n-80)
                elif n < 200:
                    samplet = 440 + 6*(n-150)
                else:
                    samplet = 740 + 16*(n-200)
                if samplet > wflen - 1:
                    break
                image_2dcharge[H, n, 0] += (fxwf[m][wflen - 1 - samplet]/40.*escale + 25)
        for m in range(len(yposition)):
            H = int((yposition[m] - ymin)/pitch)+10
            wflen = len(fywf[m])
            samplet = 0
            for n in range(255):
                if n < 80:
                    samplet = 2*n
                elif n < 150:
                    samplet = 160 + 4*(n-80)
                elif n < 200:
                    samplet = 440 + 6*(n-150)
                else:
                    samplet = 740 + 16*(n-200)
                if samplet > wflen - 1:
                    break
                image_2dcharge[H, n, 1] += (fywf[m][wflen - 1 - samplet]/40.*escale + 25)

        imsave('./images_%dmm/%s%d.jpg' % (pitch, simtype, filenum),  image_2dcharge)
    print iteration, filenum
read3dimage(3, 'background')
read3dimage(3, 'signal')
read3dimage(6, 'background')
read3dimage(6, 'signal')
read3dimage(9, 'background')
read3dimage(9, 'signal')

