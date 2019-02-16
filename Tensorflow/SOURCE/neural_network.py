# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""

import tensorflow as tf
import config

class Hidden_Layer:
    
    def __init__(self, shape, name):
        initer = tf.truncated_normal_initializer(stddev = 0.01, mean = 0)
        self.weight = tf.get_variable(name = name+"W", shape=shape, initializer = initer, dtype = tf.float32)
        self.bias = tf.get_variable(name = name+"b", shape = [shape[1]], initializer = tf.constant_initializer(0.01))
        
    def feed_forward(self, input_):
        output_ = tf.nn.relu(tf.add(tf.matmul(input_, self.weight), self.bias))
        return output_
    
class Outer_Layer:
    
    def __init__(self, shape, name):
        initer = tf.truncated_normal_initializer(stddev = 0.01, mean = 0)
        self.weight = tf.get_variable(name = name+"W", shape=shape, initializer = initer, dtype = tf.float32)
        self.bias = tf.get_variable(name = name+"b", shape = [shape[1]], initializer = tf.constant_initializer(0.01))
        
    def feed_forward(self, input_):
        output_ = tf.add(tf.matmul(input_, self.weight), self.bias)
        return output_
    
