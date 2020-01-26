#!/usr/bin/env python
# coding: utf-8
import csv
import os, glob
with open('image2dcharge_sens.csv', 'w') as csvfile:
    fieldnames = ['filename', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    filelist = glob.glob('./sens_imgs/*jpg')
    for img in filelist:
        if 'bb0n' in img:
            writer.writerow({'filename': img, 'label': '1'})
        else:
            writer.writerow({'filename': img, 'label': '0'})
    #for i in range(0, 1000000):
    #    if os.path.exists('./sens_imgs/background%d.jpg' % i):
    #        writer.writerow({'filename': './images_3mm_cl/background%d.jpg' % i, 'label': '0'})
    #for j in range(0, 1000000):
    #    if os.path.exists('./sens_imgs/signal%d.jpg' % j):
    #        writer.writerow({'filename': './images_3mm_cl/signal%d.jpg' % j, 'label': '1'})
