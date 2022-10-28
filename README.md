# ANN from scratch by A. Kha & R. Valeri

Artificial Neural Network (Multi Layers Perceptron) from scratch designed during the Biologically Inspired Computation (BIC) course, delivered by Dr W. Peng at Heriot-Watt University. The very first implementation of this work was partly inspired by [Samson Zhang's](https://www.youtube.com/c/SamsonZhangTheSalmon) youtube video [Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)](https://www.youtube.com/watch?v=w8yWXqWQYmU), which we highly recommend in order to understand a neural network training process. 

## 1. Contents

This repo contains :

• 2 Binary tabular datasets (.csv files) for breast cancer and diabetes detection

• a jupyter notebook demonstrating a prototype implementation of the MLP

• A report for the BIC coursework, summarizing the key concepts, steps and results of this work

• The lib folder, containing the main work (documented classes, methods and main test file)

## 2. Programming paradigm

Object Oriented Programming has been used to design the model class MLP, as well as auxiliary concepts (Loss, Activation, Optimizer) which enables to give a clear structure and more sense to our work.
The main difficulty of this work is the way to connect the layers in order to apply forward and backpropagation. Our solution consists in storing the weights and biases of each layer in a dictonary (each key corresponding to a certain layer), which enables to easily pass from a layer to another.
Actually, a list or even a 3D array can do the job as well (each element or first dimension element is the corresponding layer's weight/bias array). The choice of using a dictionary enables however more easily to apply some post-processing (like batchNorm) on specific layers, rather than all the layers (we batchNorm all the layers whose key is in the "batchNorm" dictionary).

## 3. Running

One could run the code, either :

• directly in the main section of the MLP.py file. The user may declare the model (like : AN = MLP()), add all the layers wanted (with the add_layer method), compile (same name method) the loss and optimizer and fit (same name method) the model with the dataset (+ target) and the hyperparameters. Two methods allow to predict the output class probabilities or the predicted classes (predict_proba & predict method). It is also possible to compute the accuracy or to plot the training curves of the model (see documentation)

• in the main.py file, which is more intuitive as the user is guided by the program. It is also possible to run this program in a python terminal, further details are presented in the 2nd README in the lib folder
