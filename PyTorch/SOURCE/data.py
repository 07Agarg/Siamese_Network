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


class DATA(Dataset):

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
        mean, std = 0.1307, 0.3081
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
        self.dataset = MNIST(root = config.DATA_DIR, train = train, transform = trans, download=True)
        if train:
            self.train_dataloader = DataLoader(self.dataset, self.batch_size, shuffle = True)
        return self.dataset

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