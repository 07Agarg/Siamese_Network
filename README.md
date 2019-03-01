# Siamese_Network

This repository contains Siamese Network for MNIST DATASET. 

### DESCRIPTION:
In Siamese Networks, we take an input image of a person and find out the encodings of that image, then we take the same network without performing any updates on weights and biases and feed this network an input image of a different person and again predict it's encodings. Now we compare these two encodings to check whether there is a similarity between the two images. These two encodings act as a latent feature representation of the images. Images with the same person have similar features/encodings. Using this, we compare and tell if the two images have the same person or not. 

### DATASET:
This network is tested on simplest MNIST DATASET, a large database of handwritten digits that is commonly used for training various image processing systems. It contains 60,000 training images and 10,000 testing images. 

The dataset is available in both Tensorflow and PyTorch. Tensorflow loads the dataset in Numpy arrays whereas PyTorch APIs loads the same dataset in Torch Tensors. 

### NETWORK DETAILS:
3-layer fully connected neural network with shared weights. 

### REQUIREMENTS:
The repository contains both PYTORCH and TENSORFLOW models . <br />
- Python 3.6 <br />
- Tensorflow 1.10.0 / PyTorch 0.2.1 <br />

### LOSS:	
Tensorflow Output: <br />
Training Data: <br />
Epoch: 48500 Loss: 0.075 <br />
Epoch: 49000 Loss: 0.095 <br />
Epoch: 49500 Loss: 0.053 <br />


PyTorch Output: <br />
Training Data: <br />
Epoch: 48500 Loss: 0.054 <br />
Epoch: 49000 Loss: 0.068 <br />
Epoch: 49500 Loss: 0.051 <br />

### RESULT:
Tensorflow:



PyTorch:



### REFERENCES:
For Visualization, I used the code of the following repo: https://github.com/ywpkwon/siamese_tf_mnist/blob/master/visualize.py






