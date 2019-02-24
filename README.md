# Siamese_Network

This repository contains Siamese Network for MNIST DATASET. 

### DESCRIPTION:
In Siamese Networks, we take an input image of a person and find out the encodings of that image, then we take the same network without performing any updates on weights and biases and feed this network an input image of a different person and again predict it's encodings. Now we compare these two encodings to check whether there is a similarity between the two images. These two encodings act as a latent feature representation of the images. Images with the same person have similar features/encodings. Using this, we compare and tell if the two images have the same person or not. 

### DATASET:
For practice, this network is tested on simplest MNIST DATASET, trying to represent the handwritten digit images using a two dimensional array. 

### NETWORK DETAILS:
3-layer fully connected neural network with shared weights. 

### REQUIREMENTS:
The repository contains both PYTORCH and TENSORFLOW models . <br />
Python 3.6 <br />
Tensorflow / PyTorch <br />

### LOSS:	
Training Data: <br />
Epoch: 48500 Loss: 0.075 <br />
Epoch: 49000 Loss: 0.095 <br />
Epoch: 49500 Loss: 0.053 <br />


### REFERENCES:
For Visualization, I used the code of the following repo: https://github.com/ywpkwon/siamese_tf_mnist/blob/master/visualize.py






