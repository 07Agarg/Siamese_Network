# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""
from tensorflow.examples.tutorials.mnist import input_data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import os
import config


class DATA():

    def __init__(self):
        #self.dir_path = os.path.join(config.DATA_DIR, dirname)
        #self.filelist = os.listdir(self.dir_path)
        #self.batch_size = config.BATCH_SIZE
        #self.size = len(self.filelist)
        self.transform = None
        self.data_index = 0
        self.dataset = None
        #self.test_data = None

    def read(self, train):
        mean, std = 0.1307, 0.3081
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
        self.dataset = MNIST(root = config.DATA_DIR, train = train, transform = trans, download=True)
        self.transform = self.dataset.transform
        #self.dataset = input_data.read_data_sets(config.DATA_DIR, one_hot = False)
        """
    def read_test(self, train):
        test_data = MNIST(root = config.DATA_DIR, train = train, )
        return test_data
        """
    def generate_train_batch(self):
   #     batch = []
   #     labels = []
   #     filelist = []
        input_1, label_1 = self.dataset.train.next_batch(config.BATCH_SIZE)
        input_2, label_2 = self.dataset.train.next_batch(config.BATCH_SIZE)
        label = (label_1 == label_2).astype('float32')
        print("label.shape " + str(label.shape))
        return input_1, input_2, label