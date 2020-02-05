#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: vbelov

import numpy as np
import torch
import torch.nn as nn
import Sniper

def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Run nEXO Detector Simulation.')
    parser.add_argument("--evtmax", '-n', type=int, default=10, help='events to be processed')
    parser.add_argument("--seed",   '-s', type=int, default=42, help='seed')
    parser.add_argument("--input",  '-i', default="sample-tmva.root", help='input')
    parser.add_argument("--output", '-o', default="sample-dnn.root", help='output')
    parser.add_argument('--pitch',  '-p', type=int, default=3, help='pad pitch')
    return parser

def save_image(image, filename):
    from PIL import Image
    Image.fromarray(image,'RGB').save(filename)

def init_tagger(filename):
    global net
    net = 'net-model'
    return None
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    import resnet_example # need to rename this
    net = resnet_example.resnet18(pretrained=False, num_classes=2, input_channels=3)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
    net = net.to(device)
    checkpoint = torch.load(filename)
    # Use trained network to tag event.
    net.load_state_dict(checkpoint['net'])

def tag_event(image_np):
    global net
    print('tag net=', net)
    return 0.
    from torchvision import transforms
    net.eval()
    #image_np = np.array(Image.open(filename), dtype=np.float32)
    image_tensor = torch.unsqueeze(transforms.functional.to_tensor(image_np), 0)
    softmax = nn.Softmax()
    with torch.no_grad():
        output = net(image_tensor.to(device))
        return softmax(output)[0][1].item()
    return 0.

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    task = Sniper.Task("task")
    task.setEvtMax(args.evtmax)
    #task.setLogLevel(0)

    # = random svc =
    import RandomSvc
    rndm = task.createSvc("RandomSvc")
    rndm.property("Seed").set(args.seed)

    # = buffer =
    import BufferMemMgr
    bufMgr = task.createSvc("BufferMemMgr")
    bufMgr.property("TimeWindow").set([0, 0]);
    #Sniper.loadDll("libSimEvent.so")
    Sniper.loadDll("libPidTmvaEvent.so")
    # ==
    import PyDataStore
    task.createSvc("PyDataStoreSvc/DataStore")
    # = rootio =
    import RootIOSvc
    ri = task.createSvc("RootInputSvc/InputSvc")
    ri.property("InputFile").set([args.input])

    ros = task.createSvc("RootOutputSvc/OutputSvc")
    ros.property("OutputStreams").set({
        #"/Event/Sim": args.output,
        #"/Event/Elec": args.output,
        "/Event/PidTmva":    args.output,
        })

    # = torch =
    #import MyAlg
    #ps = MyAlg.MyAlg("TorchAlg")
    #task.addAlg(ps)
    Sniper.loadDll("./libDnnEventTagger.so")
    tag = task.createAlg("DnnEventTagger/TorchAlg")
    tag.property("Pitch").set(args.pitch)
    #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    init_tagger('./checkpoint_%dmm_cl/ckpt.t7' % args.pitch)

    task.show()
    task.run()
