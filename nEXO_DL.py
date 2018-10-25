#!/usr/bin/env python
# coding: utf-8

#Dataset code copied from https://github.com/utkuozbulak/pytorch-custom-dataset-examples
#model code copied from https://github.com/DeepLearnPhysics/pytorch-uresnet

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data.sampler import SubsetRandomSampler
import os,sys
import shutil

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import torchvision.models as models
import argparse
import resnet_example
import traceback
from models import *
#import matplotlib.pyplot as plt

device = 'cuda' #if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epochs = 200
def save_checkpoint(state, is_best, p, filename='checkpoint.pth.tar'):
    if p>0:
        filename = "checkpoint.%dth.tar"%(p)
        torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cropandflip(npimg2d):
    imgpad  = np.zeros( (260,260), dtype=np.float32 )
    transformimg = np.zeros( (240,240, 3), dtype=np.float32 )
    flip1 = np.random.rand()
    flip2 = np.random.rand()
    randx = np.random.randint(0,10)
    randy = np.random.randint(0,10)
    for i in range(3):
        imgpad[10:240+10,10:240+10] = npimg2d[ :,:,i]
        if flip1>0.5:
            imgpad = np.flip( imgpad, 0 )
        if flip2>0.5:
            imgpad = np.flip( imgpad, 1 )
        transformimg[:,:,i] = imgpad[randx:randx+240,randy:randy+240]
    return transformimg
class nEXODatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
	# Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
	    # data augmentation
        npimg = np.array(img_as_img)
        transfomed = cropandflip(npimg)
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

# Training
def train(trainloader, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/len(trainloader), 100.*correct/total

def test(testloader, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            for m in range(outputs.size(0)):
                    print(outputs[m][1].item() - outputs[m][0].item(), targets[m].item())
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    return test_loss/len(testloader), 100.*correct/total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    transformations = transforms.Compose([transforms.ToTensor()])
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    nEXODataset = nEXODatasetFromImages('image2dcharge.csv')

    # Creating data indices for training and validation splits:
    dataset_size = len(nEXODataset)
    indices = list(range(dataset_size))
    validation_split = .2
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True
    random_seed= 42
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]


    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=200, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=400, sampler=validation_sampler)

    lr = 1.0e-3
    momentum = 0.9
    weight_decay = 1.0e-3
    batchsize = 50
    batchsize_valid = 500
    start_epoch = 0
    epochs      = 100

    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    net = resnet_example.resnet14(pretrained=False, num_classes=2, input_channels=3)
    #num_ftrs = net.fc.in_features
    #net.fc = nn.Linear(num_ftrs, 2)
    net = net.to(device)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # We use SGD
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    # define plots
    #fig,axs = plt.subplots(1,2)
    #axs[0].set_xlabel('epoch')
    #axs[0].set_ylabel('loss')
    #axs[0].set_ylim(5e-2,10.0)
    #axs[0].set_yscale("log")

    #axs[1].set_xlabel('epoch')
    #axs[1].set_ylabel('accuracy')
    #axs[1].set_ylim(0.0,100.0)

    x = np.linspace(start_epoch,start_epoch + 200,1)
    # numpy arrays for loss and accuracy
    y_train_loss = np.zeros(200)
    y_train_acc  = np.zeros(200)
    y_valid_loss = np.zeros(200)
    y_valid_acc  = np.zeros(200)
    for epoch in range(0,200):

        # set the learning rate
        adjust_learning_rate(optimizer, epoch, lr)
        iterout = "Epoch [%d]: "%(epoch)
        for param_group in optimizer.param_groups:
            iterout += "lr=%.3e"%(param_group['lr'])
            print iterout
            try:
                train_ave_loss, train_ave_acc = train(train_loader, epoch)
            except Exception,e:
                print "Error in training routine!"
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break
            print "Epoch [%d] train aveloss=%.3f aveacc=%.3f"%(epoch,train_ave_loss,train_ave_acc)
            y_train_loss[epoch] = train_ave_loss
            y_train_acc[epoch]  = train_ave_acc

            # evaluate on validationset
            try:
                valid_loss,prec1 = test(validation_loader, epoch)
            except Exception,e:
                print "Error in validation routine!"
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break
            print "Test[%d]:Result* Prec@1 %.3f\tLoss %.3f"%(epoch,prec1,valid_loss)
            y_valid_loss[epoch] = valid_loss
            y_valid_acc[epoch]  = prec1
            # plot up to current iteration
            #axs[0].plot(x[:epoch+1],y_train_loss[:epoch+1],'b')
            #axs[0].plot(x[:epoch+1],y_valid_loss[:epoch+1],'r')
            # plot up to current iteration
            #axs[1].plot(x[:epoch+1],y_train_acc[:epoch+1],'b')
            #axs[1].plot(x[:epoch+1],y_valid_acc[:epoch+1],'r')
        print("Train loss", y_train_loss)
        print("Train acc", y_train_acc)
        print("Valid loss", y_valid_loss)
        print("Valid acc", y_valid_acc)

            #fig.canvas.draw()
            #plt.savefig('acc_loss.jpg')
    #for epoch in range(start_epoch, start_epoch+200):
    #    train(train_loader, epoch)
       # test(validation_loader, epoch)
