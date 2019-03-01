# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""
#import os
import data
import model
import torch
import utils 

if __name__ == "__main__":
    # READ DATA
    data_ = data.DATA()
    train_dataset = data_.read(True)
    print("Train Data Loaded")
    # BUILD MODEL
    net = model.MODEL()
    print("Model Initialized")
    modeloperator = model.Operators(net)    
    print("Model Built")
    # TRAIN MODEL
    #modeloperator.train(data_)
    print("Model Trained")
    # TEST MODEL
    test_dataset = data_.read(False)
    print("test_dataset shape " + str(len(test_dataset)))
    print("Test Data Loaded")
    test_data_x = test_dataset.test_data.type(torch.FloatTensor)
    test_data_y = test_dataset.test_labels
    #test_data_x, test_data_y = test_dataset.test_data, test_dataset.test_labels
    print("type test_data_x " +str(type(test_data_x)) )
    #test_data_x, test_data_y = data_.read_test()
    modeloperator.test(test_data_x)
    print("Embedding File Created")
    utils.visualize(test_data_x.numpy(), test_data_y.numpy())
    #utils.visualize(test_data_x, test_data_y)

