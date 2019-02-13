# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""

import os

# DIRECTORY INFORMATION
DATASET = "Dogs"
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')
LOG_DIR = os.path.join(ROOT_DIR, 'LOGS/')

TRAIN_DIR = "train"
TEST_DIR = "test"

# DATA INFORMATION
IMAGE_SIZE = 28*28
BATCH_SIZE = 128

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
USE_PRETRAINED = False
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
LAYERS = [(IMAGE_SIZE, 1024), (1024, 1024), (1024, 2)]

