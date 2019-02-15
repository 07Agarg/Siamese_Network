# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import os
import config


class DATA():

    def __init__(self, dirname):
        self.dir_path = os.path.join(config.DATA_DIR, dirname)
        self.filelist = os.listdir(self.dir_path)
        self.batch_size = config.BATCH_SIZE
        self.size = len(self.filelist)
        self.data_index = 0
        self.dataset = None
        #self.test_data = None

    def read(self, filename):
        self.dataset = input_data.read_data_sets('MNIST_data', one_hot = False)
        
    def read_test(self):
        test_data = self.dataset.test.images
        return test_data
        
    def generate_train_batch(self):
   #     batch = []
   #     labels = []
   #     filelist = []
        input_1, label_1 = self.dataset.train.next_batch(config.BATCH_SIZE)
        input_2, label_2 = self.dataset.train.next_batch(config.BATCH_SIZE)
        label = (label_1 == label_2).astype('float32')
        return input_1, input_2, label
        
        """for i in range(self.batch_size):
            filename = os.path.join(config.DATA_DIR, self.dir_path, self.filelist[self.data_index])
            filelist.append(self.filelist[self.data_index])
            greyimg, colorimg = self.read_img(filename)
            batch.append(greyimg)
            labels.append(colorimg)
            self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch)/255
        labels = np.asarray(labels)/255
        return batch, labels, filelist
"""