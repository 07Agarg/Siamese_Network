# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import os
import config
from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataset import Dataset
from tensorflow.examples.tutorials.mnist import input_data
"""
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import config


class DATA():

    def __init__(self):
        self.data_index = 0
        self.dataset = None

    def read(self):
        self.dataset = input_data.read_data_sets(config.DATA_DIR, one_hot = False)
        
    def read_test(self):
        test_data_x = self.dataset.test.images
        test_data_y = self.dataset.test.labels
        return test_data_x, test_data_y
        
    def generate_batch(self):
        input_1, label_1 = self.dataset.train.next_batch(config.BATCH_SIZE)
        input_2, label_2 = self.dataset.train.next_batch(config.BATCH_SIZE)
        label = (label_1 == label_2).astype('float32')
        return input_1, input_2, label




"""
class DATA( ):

    def __init__(self):
        #self.dir_path = os.path.join(config.DATA_DIR, dirname)
        #self.filelist = os.listdir(self.dir_path)
        #self.size = len(self.filelist)
        self.batch_size = config.BATCH_SIZE
        self.transform = None
        self.data_index = 0
        self.dataset = None
        self.train_dataloader = None
        self.test_dataloader = None
        #self.test_data = None

    def read(self, train):
       # mean, std = 0.1307, 0.3081
        #trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
        trans = transforms.Compose([transforms.ToTensor()])
        self.dataset = MNIST(root = config.DATA_DIR, train = train, transform = trans, download=True)
        if train:
            self.train_dataloader = DataLoader(self.dataset, self.batch_size, shuffle = True)
        return self.dataset

    def read_test(self):
        dataset = input_data.read_data_sets(config.DATA_DIR, one_hot = False)   
        test_data_x = dataset.test.images
        test_data_y = dataset.test.labels
        return test_data_x, test_data_y
    
    def generate_batch(self):
        train_iter = iter(self.train_dataloader)
        input_1, label_1 = next(train_iter)
        input_2, label_2 = next(train_iter)
        input_1 = input_1.reshape(input_1.size()[0], -1)
        input_2 = input_2.reshape(input_2.size()[0], -1)
        np_label_1 = label_1.numpy()
        np_label_2 = label_2.numpy()
        label = (np_label_1 == np_label_2).astype('float32')
        return input_1, input_2, label
        