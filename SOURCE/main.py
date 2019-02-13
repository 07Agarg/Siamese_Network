# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import data
import model
import config
import datetime
import tensorflow as tf
import utils 

if __name__ == "__main__":
    with open(os.path.join(config.LOG_DIR, str(datetime.datetime.now().strftime("%Y%m%d")) + "_" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".txt"), "w") as log:
       # log.write(str(datetime.datetime.now()) + "\n")
       # log.write("Use Pretrained Weights: " + str(config.USE_PRETRAINED) + "\n")
       # log.write("Pretrained Model: " + config.PRETRAINED + "\n")
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
        test_data = data_.read_test()
        print("Test Data Loaded")
        model.test(test_data)
        print("Embedding File Created")
        utils.visualize()


