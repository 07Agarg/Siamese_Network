# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""
#import os
import data
import model
#import config
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
    modeloperator.train(data_)
    print("Model Trained")
    # TEST MODEL
    test_dataset = data_.read(False)
    print("Test Data Loaded")
    test_data_x, test_data_y = test_dataset.test_data, test_dataset.test_labels
    modeloperator.test(test_dataset.test_data)
    print("Embedding File Created")
    utils.visualize(test_data_x.numpy(), test_data_y.numpy())

