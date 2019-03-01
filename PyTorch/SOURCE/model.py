# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""


import torch 
import torch.nn as nn
from torch.autograd import Variable
import config
import os
#import data
torch.manual_seed(config.SEED)
    

class MODEL(nn.Module):
    
    def __init__(self):
        super(MODEL, self).__init__()
        self.HiddenLayer_1 = nn.Linear(config.SHAPE[0][0], config.SHAPE[0][1])
        self.HiddenLayer_2 = nn.Linear(config.SHAPE[1][0], config.SHAPE[1][1])
        self.OutputLayer = nn.Linear(config.SHAPE[2][0], config.SHAPE[2][1])
        
    def forward_once(self, X):
        output = nn.functional.relu(self.HiddenLayer_1(X))
        output = nn.functional.relu(self.HiddenLayer_2(output))
        output = self.OutputLayer(output)
        return output
    
    def forward(self, X1, X2):
        out_1 = self.forward_once(X1)
        out_2 = self.forward_once(X2)
        return out_1, out_2
        
class ContrastiveLoss(torch.nn.Module):
    
    def __init__(self, margin = 5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6
        
    def forward(self, out_1, out_2, Y):
        euclidean_distance = nn.functional.pairwise_distance(out_1, out_2)
        loss_contrastive = torch.mean((Y) * torch.pow(euclidean_distance, 2) +
                                      (1 - Y) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        """
        distances = (out_1 - out_2).pow(2).sum(1)  # squared distances
        losses = 0.5 * (Y.float() * distances + (1 + -1 * Y).float() * nn.functional.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        
        eucd2 = torch.pow(out_1 - out_2, 2)
        eucd_pos = torch.sum(eucd2, 1)
        eucd = torch.sqrt(eucd_pos+1e-6)
        eucd_neg = torch.pow(torch.clamp(self.margin - eucd, min = 0.0), 2)
        loss_pos = torch.mul(Y, eucd_pos)
        loss_neg = torch.mul((1 - Y), eucd_neg)
        loss = torch.mean(loss_pos + loss_neg)
        
        eucd = nn.functional.pairwise_distance(out_1, out_2)
        eucd_pos = torch.pow(eucd, 2)
        eucd_neg = torch.pow(torch.max(0.0, (self.margin - eucd)), 2)
        loss_pos = torch.mul(Y, eucd_pos)
        loss_neg = torch.mul((1 - Y), eucd_neg)
        loss = torch.mean(torch.add(loss_pos, loss_neg))
        """
        return loss_contrastive

class Operators():
    
    def __init__(self, net):
        self.net = net
        self.loss = ContrastiveLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr = config.LEARNING_RATE)
        
    def train(self, data):
        #train_loader = torch.utils.data.DataLoader(dataset = data, batch_size = config.BATCH_SIZE, shuffle = True)
        
        for epoch in range(config.NUM_EPOCHS):
            input_1, input_2, out = data.generate_batch()
            X_1 = Variable(torch.Tensor(input_1).float())
            X_2 = Variable(torch.Tensor(input_2).float())
            Y = Variable(torch.Tensor(out).float())
            self.optimizer.zero_grad()
            out_1, out_2 = self.net.forward(X_1, X_2)
            loss_val = self.loss.forward(out_1, out_2, Y)
            loss_val.backward()
            self.optimizer.step()
            if epoch % 500 == 0:
                print('Epoch: %d Loss: %.3f' % (epoch, loss_val))
                #print("Epoch: ", (epoch + 1), "loss = ", "{:.3f}".format(_loss.data[0]))
        torch.save(self.net, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pt"))
        
    def test(self, dataX):
        self.net = torch.load(os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pt"))
        self.net.eval()
        print("dataX type " +str(type(dataX)))
        #input_1 = Variable(torch.Tensor(dataX).float())
        #print("input_1 type " + str(type(input_1)))
        dataX = dataX.reshape(dataX.size()[0], -1)
        #print("dataX type " + str(type(dataX)))
        #dataX = Variable(torch.Tensor(dataX).float())
        #dataX = Variable(torch.Tensor(dataX).float())
        print("dataX shape " + str(dataX.shape))
        output = self.net.forward_once(dataX)
        output.data.numpy().tofile(os.path.join(config.OUT_DIR, 'embed.txt'))