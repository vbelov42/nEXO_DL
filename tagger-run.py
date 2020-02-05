#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: vbelov

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
    import DnnEventTagger
    tag = task.createAlg("DnnEventTagger/TorchAlg")
    tag.property("Pitch").set(args.pitch)

    task.show()
    task.run()
