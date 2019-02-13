# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""
import os
import tensorflow as tf
import config
import neural_network

class MODEL():
    
    def __init__(self):
        self.inputs_1 = tf.placeholder(shape = [None, config.IMAGE_SIZE], dtype = tf.float32)    
        self.inputs_2 = tf.placeholder(shape = [None, config.IMAGE_SIZE], dtype = tf.float32)    
        self.labels = tf.placeholder(shape = [], dtype = tf.float32)
        self.output_1 = None
        self.output_2 = None
        self.loss = None
        
    def build(self, input_):
        for i in range(len(config.LAYERS)-1):
            hidden = neural_network.Hidden_Layer(config.LAYERS[i])
            input_ = hidden.feed_forward(input_)
        
        outer = neural_network.Outer_Layer(config.LAYERS[-1])
        output_ = outer.feed_forward(input_)
        return output_
       
    def loss_fun(self, margin = 5):
        labels = self.labels
        eucd2 = tf.pow(tf.subtract(self.output_1, self.output_2), 2)
        eucd_pos = tf.reduce_sum(eucd2, 1, name = "eucd_pos")
        eucd = tf.sqrt(eucd2 + 1e-6, name = "eucd")
        eucd_neg = tf.pow(tf.maximum(tf.subtract(margin, eucd), 0), 2, name = "eucd_neg")
        loss_pos = tf.multiply(labels, eucd_pos, name = "pos_contrastive_loss")
        loss_neg = tf.multiply((1-labels), eucd_neg, name = "neg_conrastive_loss")
        loss = tf.reduce_mean(tf.add(loss_pos, loss_neg), name = "contrastive_loss")
        return loss
    
    def train(self, data):
        self.loss = self.loss_fun()
        optimizer = tf.train.GradientDescentOptimizer(config.LEARNING_RATE).minimize(self.loss)
        saver = tf.train.Saver()
        with tf.Session() as session:
            init = tf.global_variables_initializer()
            session.run(init)
            print("Variables initialized.... ")
            for epoch in range(config.NUM_EPOCHS):
                cost = 0
                total_batch = int(data.size/config.BATCH_SIZE)
                for i in range(total_batch):
                    batch_X1, batch_X2, batch_label = data.generate_train_batch()
                    feed_dict = {self.inputs_1: batch_X1, self.inputs_2: batch_X2, self.labels: batch_label}
                    loss_val, _ = session.run([self.loss, optimizer], feed_dict = feed_dict)
                    print("batch: ", i, "loss: ", loss_val)
                    cost += (loss_val/total_batch)
                print("Epoch: ", (epoch+1), "cost = ", "{:.3f}".format(cost))
                
            saver.save(session, os.path.join(config.MODEL_DIR, "model.ckpt"))
            
    def test(self, input_1):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(config.MODEL_DIR, "model.ckpt"))            
            feed_dict = {self.inputs_1: input_1}
            output = self.session.run(self.output_1, feed_dict = feed_dict)
            output.tofile('embed.txt')
        