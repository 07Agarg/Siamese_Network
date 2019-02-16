# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""
import os
import data
import model
import tensorflow as tf
import utils 

if __name__ == "__main__":
    # READ DATA
    data_ = data.DATA()
    data_.read()
    print("Train Data Loaded")
    # BUILD MODEL
    model = model.MODEL()
    print("Model Initialized")
        
    with tf.variable_scope("siamese") as scope:
        model.output_1 = model.build(model.inputs_1)
        scope.reuse_variables()
        model.output_2 = model.build(model.inputs_2)
            
    print("Model Built")
    # TRAIN MODEL
    model.train(data_)
    print("Model Trained")
    # TEST MODEL
    test_data_x, test_data_y = data_.read_test()
    print("Test Data Loaded")
    model.test(test_data_x)
    print("Embedding File Created")
    utils.visualize(test_data_x, test_data_y)
    print("Visualization Created")


