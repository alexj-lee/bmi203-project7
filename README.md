![BuildStatus](https://github.com/alexj-lee/bmi203-project7/actions/workflows/test.yml/badge.svg)

See https://alexj-lee.github.io/bmi203-project7/writeup for html version of the jupyter notebook

# Project 7
Fully Connected Neural Network


# Assigment

## Overview
In this assignment, you will try out your machine learning skills by first implementing a neural network
class from (almost) scratch. You will then apply your class to implement both

**(1) a simple 64x16x64 autoencoder**

**(2) a classifier for transcription factor binding sites**

In this assigment you will begin by finishing the API for generating fully connected neural networks from scratch.
Next, you will make a jupyter notebook where you will implement your 64x16x64 neural network and your 
classifier for transcription factor binding sites.


## Step 1: Finish the neural network API
TODO:
* Finish all incomplete methods that have a pass statement in the NeuralNetwork class in the nn.py file
* Finish the 'one_hot_encode_seqs' function in the preprocess.py file
* Finish the 'sample_seqs' function in the preprocess.py file


## Step 2: Generate your Autoencoder
### Background
Autoencoders are a type of neural network architecture that takes in an input, encodes that information
into a lower-dimensional latent space via 'encoding' layer(s) and then attempts to reconstruct the intial
input via 'decoding' layer(s). Autoencoders are most often used as a dimensionality reduction technique.

### Your task
Here you will train a 64x16x64 autoencoder. All of the following work should be done in a jupyter notebook.

TODO:
* Generate an instance of your NeuralNetwork class for the 64x16x64 autoencoder
* Read in toy dataset 'digits' from sklearn using digits = sklearn.datasets.load_digits()
* Split the digits dataset into a train and a validation set
* Train your autoencoder on the train split of your digits dataset
* Plot your training and validation loss per epoch
* Explain in your jupyter notebook why you chose the hyperparameter values that you did
* Quantify your average reconstruction error over the dataset.


## Step 3: Generate your Transcription Factor Classifier
### Background
Transcription factors are proteins that bind DNA at promoters to drive gene expression. 
Most preferentially bind specific patterns of sequence, while ignoring others. 
Traditional methods to determine this pattern (called a motif) have assumed that binding 
sites in the genome are all independent. However, there are cases where people have identified motifs where
positional interdependencies exist.

### Your task
Here you will implement a multilayer fully connected neural network using your NeuralNetwork class
capable of accurately predicting whether a short DNA sequence is a binding site for the 
yeast transcription factor Rap1. Note that the training data is incredibly imbalanced as
there are much fewer positive sequences than negative sequences. In order to overcome this
you will need to implement a sampling scheme to ensure that class imbalance does not affect
your training.

TODO:
* Use the 'read_text_file' function from preprocess.py to read in the 137 positive Rap1 motif examples
* Use the 'read_fasta_file' function to read in all the negative examples from all 1kb upstream in yeast.
* Implement a sampling scheme in the 'sample_seq' function in the preprocess.py file
* Explain in your jupyter notebook why chose the sampling scheme that you did.
* Generate a training and a validation set for training your classifier.
* One hot encode your training and validation sets using your implementation of the 'one_hot_encode_seqs' function in the preprocess.py file
* Train your neural network!
* Explain your choice of loss function in the jupyter notebook
* Explain your choice of hyperparameters in the jupyter notebook
* Plot the training and validation loss curves per epoch
* Print out the accuracy of your classifier on your validation dataset


# Grading (50 points total)

## Implementation of API (15 points)
* Proper implementation of NeuralNetwork class (13 points)
* Proper implementation of 'one_hot_encode_seqs' function (1 point)
* Proper implementation of 'sample_seqs' function (1 point)

## Autoencoder (10 points)
* Read in dataset and generate train and validation splits (2 points)
* Successfully train your autoencoder (4 points)
* Plots of training and validation loss (2 points)
* Explanation of hyperparameters (1 point)
* Quantification of reconstruction error (1 point)

## Transcription Factor Classifier (15 points)
* Correctly read in all data (2 points)
* Explanation of your sampling scheme (2 points)
* Proper generation of a training set and a validation set (2 point)
* Successfully train your classifeir (4 points)
* Explain the choice of your loss function in the jupyter notebook (2 points)
* Plots of training and validation loss (2 points)
* Print out accuracy of the classifier on the testing set (1 point)

## Testing (7 points)
Proper unit tests for:
* forward method (1 point)
* _single_forward method (1 point)
* _single_backprop method (1 point)
* predict method (1 point)
* binary_cross_entropy loss method (0.5 points)
* binary_cross_entropy_backprop method (0.5 points)
* mean_squared_error loss function (0.5 points)
* mean_squared_error_backprop (0.5 points)
* one_hot_encode_seqs function (0.5 points)
* sample_seqs function (0.5 points)

## Packaging (3 points)
* pip installable (1 point)
* github actions (installing + testing) (2 points)


