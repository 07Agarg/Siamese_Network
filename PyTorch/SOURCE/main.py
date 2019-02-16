# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""
import os
import data
import model
import config
import utils 

if __name__ == "__main__":
    # READ DATA
    data_ = data.DATA()
    train_dataset = data.read(True)
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
    test_dataset = data.read(False)
    print("Test Data Loaded")
    model.test(test_data)
    print("Embedding File Created")
    utils.visualize()
