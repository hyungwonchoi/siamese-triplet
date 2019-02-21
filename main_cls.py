from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torchvision
from datasets import TripletMNIST, TripletOmnious
import torchvision.datasets as dset

from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
#import torchvision.transforms as transforms
import transforms
from PIL import Image

import os
import argparse

from models import *
from torch.autograd import Variable
from tensorboardX import SummaryWriter





parser = argparse.ArgumentParser(description='PyTorch Triplet network for classification/embedding, on MNIST/fashion MNIST')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--batch_size', default=32, type=int, help='batchSize')
parser.add_argument('--nchannel', default=1, type=int, help='number of channel of data')
parser.add_argument('--nfeat', default=2, type=int, help='number of feature')
parser.add_argument('--dataset', default='fashionMnist', help='dataset')
parser.add_argument('--type', default='item', help='dataset type for omnious db')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
parser.add_argument('--triplet', default=1e-1, type=float, help='triplet loss ratio (default=0.1)')

parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--epoch', default=20, type=int, help='number of epochs')
parser.add_argument('--ntry', default=1, type=int, help='ntry')
parser.add_argument('--log_interval', default=500, type=int, help='log interval')
parser.add_argument('--save_embedding', default=True, type=bool, help='Save embeddings')
parser.add_argument('--dataroot', required=True, help='Path to the dataset')
args = parser.parse_args()

torch.manual_seed(args.seed)

writer = SummaryWriter("./visual/{}/{}/nfeat_{}_triplet_{}_lr_{}_ntry{}".format(args.dataset, args.type, args.nfeat, args.triplet, args.lr, args.ntry))

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels



mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

cuda = torch.cuda.is_available()

if args.dataset =='Mnist':
    print('Using MNIST')

    mean, std = 0.1307, 0.3081
    train_dataset = MNIST('../data/MNIST', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((mean,), (std,))
                                 ]))
    test_dataset = MNIST('../data/MNIST', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))
                                ]))
    
    
    # Set up data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    n_classes = 10

elif args.dataset =='fashionMnist':
    print('Using fashion MNIST')
    mean, std = 0.28604059698879553, 0.35302424451492237

    train_dataset = FashionMNIST('../data/FashionMNIST', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((mean,), (std,))
                                 ]))
    test_dataset = FashionMNIST('../data/FashionMNIST', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))
                                ]))

    
    # Set up data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    n_classes = 10
    mnist_classes = fashion_mnist_classes    
       
    

elif args.dataset =='omni':
    scale_resize = 256
    resize = 224
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))   

    print("Using random crop... ")
    transform_train = transforms.Compose([
        transforms.ScaleLong(scale_resize,interpolation=Image.ANTIALIAS),
        transforms.RandomCrop(resize),            
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])        
            
    transform_test = transforms.Compose([
            transforms.ScaleLong(scale_resize,interpolation=Image.ANTIALIAS),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            normalize,
        ])

    
    kwargs = {'num_workers': 4, 'pin_memory': True}

    traindir = os.path.join(args.dataroot, 'train_balanced')    
    valdir = os.path.join(args.dataroot,'test_balanced')

    train_dataset = dset.ImageFolder(root=traindir,transform=transform_train)
    val_dataset = dset.ImageFolder(root=valdir,transform=transform_test)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)        
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)  
    
    n_classes = len(train_dataset.classes)

    train_labels = []
    test_labels = []
    for i in range(len(train_dataset.imgs)):
        train_labels.append(train_dataset.imgs[i][1])
    for i in range(len(val_dataset.imgs)):
        test_labels.append(val_dataset.imgs[i][1])        
        
    
# Set up the network and training parameters
from networks import EmbeddingNet, ClassificationNet, EmbeddingNet_ResNet
from metrics import AccumulatedAccuracyMetric

embedding_net = EmbeddingNet_ResNet(args.nfeat, args.nchannel)
model = ClassificationNet(embedding_net, n_classes=n_classes, nfeat=args.nfeat)
if cuda:
    model.cuda()
loss_fn = torch.nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

clsonly = True

if args.save_embedding == True:
    #triplet_test_dataset = TripletMNIST(test_dataset)
    triplet_test_dataset = TripletOmnious(val_dataset, test_labels, istrain=False)
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    fit(clsonly, writer, args, train_loader, test_loader, model, loss_fn, optimizer, scheduler, args.epoch, cuda, args.log_interval, metrics=[AccumulatedAccuracyMetric()], val_loader_emb=triplet_test_loader)

else:
    fit(clsonly, writer, args, train_loader, test_loader, model, loss_fn, optimizer, scheduler, args.epoch, cuda, args.log_interval, metrics=[AccumulatedAccuracyMetric()])

#train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
#plot_embeddings(train_embeddings_baseline, train_labels_baseline)
#val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
#plot_embeddings(val_embeddings_baseline, val_labels_baseline)


#writer.add_embedding(val_embeddings_baseline, metadata=val_labels_baseline, global_step=1)
writer.close()