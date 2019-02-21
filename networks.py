import torch.nn as nn
import torch.nn.functional as F
from models.resnet import *

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
class EmbeddingNet_ResNet(nn.Module):
    def __init__(self, numfeat=128, nchannel=3):
        super(EmbeddingNet_ResNet, self).__init__()
        self.resnet = resnet50(pretrained=True, numfeat=numfeat)
        #self.convnet = nn.Sequential(nn.Conv2d(nchannel, 64, 5), nn.ReLU(),
        #                             nn.MaxPool2d(2, stride=2),
        #                             nn.Conv2d(64, 128, 3), nn.ReLU(),
        #                             nn.MaxPool2d(2, stride=2),
        #                             nn.Conv2d(128, 256, 3), nn.ReLU(),
        #                             nn.MaxPool2d(2, stride=2))

        #self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
        #                        nn.ReLU(),
        #                        nn.Linear(256, 256),
        #                        nn.ReLU(),
        #                        nn.Linear(256, nfeat)
        #                        )

    def forward(self, x):
        output = self.resnet(x)
        #output = self.convnet(x)
        #output = output.view(output.size()[0], -1)
        #output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNet(nn.Module):
    def __init__(self, nfeat=2, nchannel=1):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(nchannel, 64, 5), nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 128, 3), nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(128, 256, 3), nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, nfeat)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes, nfeat):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(nfeat, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))



class ClassificationNet_normsoftmax(nn.Module):
    def __init__(self, embedding_net, n_classes, nfeat, temp=0.05):
        super(ClassificationNet_normsoftmax, self).__init__()
        self.temp = temp
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(nfeat, n_classes, bias=False)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        
        # Layer norm
        lnorm = nn.LayerNorm(output.size()[1:], elementwise_affine=False)
        output = lnorm(output)
        
        # L2 normalize input before softmax
        output = F.normalize(output, p=2, dim=1)
        
        # L2 normalize weights of linear layer
        self.fc1.weights = F.normalize(self.fc1.weight, p=2)
        
        # forward linear
        output = self.fc1(output)
        
        scores = F.log_softmax(output/self.temp, dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))
    
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
