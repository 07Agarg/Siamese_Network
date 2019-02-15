# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""

import tensorflow as tf
import config

class Hidden_Layer:
    def __init__(self, shape):
        self.weight = tf.get_variable(tf.truncated_normal_initializer(shape=shape ,stddev=0.01, mean = 0))
        self.bias = tf.get_variable(tf.constant(0.1, shape = [shape[1]]))
        
    def feed_forward(self, input_):
        output_ = tf.nn.relu(tf.add(tf.matmul(input_, self.weight), self.bias))
        return output_
    
class Outer_Layer:
    def __init__(self, shape):
        self.weight = tf.get_variable(tf.random_normal(shape=shape,stddev=0.01, mean = 0))
        self.bias = tf.get_variable(tf.constant(0.1, shape=[shape[1]]))
        
    def feed_forward(self, input_):
        output_ = tf.add(tf.matmul(input_, self.weight), self.bias)
        return output_
    
