This repository concerns the implementation of a Multilayer Perceptron from scratch
(without using a machine learning library) designed during a coursework of the
Biologically Inspired Computation course at Heriot-Watt University.

Authors : Alexandre Kha and Raphaël Valeri
Date : 27th October 2022

To run the code you can execute the following command
    python main.py
    - this will allow you to design the Artifical Network as you want (by defining the hyperparameters)

To access the documentation of this execution you can run the following command:
    python main.py -h

If you want to use the default architecture and hyperparameters you can run with the option -d
    python main.py -d

Other options as available:
    -p to plot the training curve
    -b to add batch normalization between layers
    -k with a number to run k-cross validation after the training