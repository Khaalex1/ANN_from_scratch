import numpy as np
import pandas as pd
from lib import *
import argparse
import MinMax
import MLP

def main():
    """
    Main function to use the implementation of the MLP
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--default_hyperparameters", help="Use the default value for the hyperparameters", action="store_true")
    parser.add_argument("-b", "--batch_normalization", help="Use batch-normalization after each layer (except the output layer)", action="store_true")
    parser.add_argument("-k", "--cross_validation", type=int, default=0, help="Use cross_validation with the number of fold specified by this option")
    parser.add_argument("-p", "--plot", help="Plot the training curve of the accuracy and loss of the model during training", action="store_true")

    args = parser.parse_args()
    if args.default_hyperparameters:
        #Default hyperparameters
        print("Default hyperparameters will be used")
        nodes = [64, 32, 2]
        activations = ['relu', 'relu', 'softmax']
        loss = 'cross_entropy'
        optimizer = 'minibatch'
        l_rate = 0.01
        BATCH_SIZE = 8
        EPOCHS = 125
        print('---'*10)
        print('Layers : \n')
        print('   Neurons : {}'.format(nodes))
        print('   Activations functions : {}'.format(activations))
        print('---' * 10)
        print('Training hyperparameters ')
        print('Loss function : {}'.format(loss))
        print('Optimizer : {}'.format(optimizer))
        print('Learning rate : {}'.format(l_rate))
        print('Batch-size : {}'.format(BATCH_SIZE))
        print('Epochs : {}'.format(EPOCHS))
    else:
        #Ask the user to define the hyperparameters
        nodes, activations, loss, optimizer, l_rate, BATCH_SIZE, EPOCHS = get_hyperparameters()
    #Load the dataset into features matrix for training and testing
    X_train, Y_train, X_test, Y_test, nb_feat = load_dataset()
    #Build the Multilayer Perceptron with the defined hyperparameters
    model = MLP.MLP()
    for i in range(len(nodes)):
        if i == 0:
            model.add_layer(nb_nodes=nodes[i], feat_size=nb_feat, activation=activations[i])
        else:
            model.add_layer(nb_nodes=nodes[i], activation=activations[i])
        if args.batch_normalization and (i != len(nodes)-1):
            model.BatchNormalization()
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(X_train, Y_train, BATCH_SIZE, EPOCHS, l=l_rate)
    # Evaluation with the test dataset
    print("---"*10)
    print("Evaluation of the model on the test dataset :")
    y_pred = model.predict(X_test)
    acc_test = MLP.accuracy(y_pred, Y_test)
    print('Test accuracy : {}'.format(np.round(acc_test, 4)))
    # Check the options of plot and cross-validation
    if args.plot:
        model.training_curve()
    if args.cross_validation:
        print('---'*10)
        print('{}-fold cross_validation :'.format(args.cross_validation))
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
        acc, locc = model.Kfold_simulation(X, Y, args.cross_validation, BATCH_SIZE, EPOCHS, l_rate,)


def load_dataset():
    """
    Load the breast cancer dataset and pre-process it with Min-Max Scale
    :return:
    """
    # Load the dataset
    data = pd.read_csv('../wdbc.csv')
    # Add columns
    data.columns = [i for i in range(32)]
    # Replace the categorical labels by int
    d = {'B': 0, 'M': 1}
    label_col = data[1].replace(d)
    # Delete the id and the categorical labels and add the numerical labels at the end
    data.pop(1)
    data.pop(0)
    data['label'] = label_col

    print('---'*10)
    print('The MLP will use the following dataset of breast cancer data')
    print(data.head())
    print('Number of features : {}'.format(data.shape[1] -1))
    print('Number of samples : {}'.format(data.shape[0]))
    print('---'*10)
    nb_split = ask_request('Please enter the fraction of the data you want to keep for the training process (the other for test) :', 'Please enter a fraction between 0.0 (not included) and 1.0', type='float', string_range=None)

    #Shuffle the dataset and split into train and test dataset
    data = data.sample(frac=1)
    data = data.to_numpy()
    nb_samp, nb_feat = data.shape
    nb_feat -= 1
    nb_train = int(nb_split * nb_samp)

    X_train = data[:nb_train, :nb_feat]
    X_test = data[nb_train:, :nb_feat]

    Y_train = data[:nb_train, nb_feat].astype('int')
    Y_test = data[nb_train:, nb_feat].astype('int')

    # MinMax normalization
    MM = MinMax.MinMax()
    X_train = MM.fit_transform(X_train)
    X_test = MM.transform(X_test)
    return X_train, Y_train, X_test, Y_test, nb_feat


def get_hyperparameters():
    """
    Ask the user to set the hyperparamameters of the Multilayer perceptron
    :return:
    """
    print('Enter the hyperparameters you want to use')
    print('---' * 10)
    print('Number of hidden layers :')
    nb_layers = int(input())
    while nb_layers <= 0:
        print('Please enter a positive number')
        nb_layers = int(input())
    nodes = []
    activations = []
    for n in range(1, nb_layers + 1):
        nodes_n = ask_request('Number of neurons in layer {} :'.format(n), 'Please enter a positive number', type='int',
                              string_range=None)
        nodes.append(nodes_n)
        activations_p = ['relu', 'tanh', 'sigmoid']
        activation_n = ask_request('Activation function in layer {} :'.format(n),
                                   'Please enter a valid activation function ({})'.format(activations_p), type='string',
                                   string_range=activations_p)
        activations.append(activation_n)
    n_output = ask_request('Number of neurons in the output layer :', 'Please enter a positive number', type='int',
                           string_range=None)
    nodes.append(n_output)
    if n_output >= 2:
        possible_output_act = ['softmax']
    else:
        possible_output_act = ['sigmoid']
    act_output = ask_request('Activation function in the output layer :',
                             'Please enter a valid activation function in ({})'.format(possible_output_act),
                             type='string', string_range=possible_output_act)
    activations.append(act_output)
    print('---' * 10)
    print('You have chosen this architecture : ')
    print('Layers : \n')
    print('   Neurons : {}'.format(nodes))
    print('   Activations functions : {}'.format(activations))
    print('---' * 10)
    print('Training hyperparameters ')
    if n_output >= 2:
        loss_functions = ['cross_entropy']
    else:
        loss_functions = ['binary_cross_entropy', 'mse', 'abs']
    optimizers_p = ['sgd', 'batch', 'minibatch', 'adam', 'rmsprop']
    loss = ask_request('Loss function :', 'Please enter a valid loss function ({})'.format(loss_functions),
                       type='string', string_range=loss_functions)
    optimizer = ask_request('Optimizer :', 'Please enter a valid optimizer ({})'.format(optimizers_p), type='string',
                            string_range=optimizers_p)
    l_rate = ask_request('Learning rate :', 'Please enter a positive number', type='float', string_range=None)
    if optimizer == 'sgd':
        BATCH_SIZE = 1
    elif optimizer == 'batch':
        BATCH_SIZE = 100  # random value it will not be used
    else:
        BATCH_SIZE = ask_request('Batch-size :', 'Please enter a strict positive number ', type='int',
                                 string_range=None)
    EPOCHS = ask_request('Number of epoch :', 'Please enter a strict positive number', type='int', string_range=None)
    print('---' * 10)
    print('You have chosen the following training hyperparameters :')
    print('Loss function : {}'.format(loss))
    print('Optimizer : {}'.format(optimizer))
    print('Learning rate : {}'.format(l_rate))
    if optimizer not in ['sgd', 'batch']:
        print('Batch-size : {}'.format(BATCH_SIZE))
    print('Epochs : {}'.format(EPOCHS))
    return nodes, activations, loss, optimizer, l_rate, BATCH_SIZE, EPOCHS



def ask_request(message, error_message, type, string_range,  int_range=None):
    """
    Ask the user to enter a value
    :param message: Message to display to the user
    :param error_message: Error message if the value entered do not follow the requierements
    :param type: type of the value ('int', 'float' or 'string')
    :param string_range: ranges of possible answer if type='string'
    :param int_range: ranges of possibles answer if type='int' and the requierements is not to be just positive
    :return: the value entered by the user
    """
    print(message)
    if type == 'int':
        value = int(input())
        if int_range is not None:
            while value not in int_range:
                print(error_message)
                value = int(input())
        else:
            while value <= 0:
                print(error_message)
                value = int(input())
        return value
    if type == 'float':
        value = float(input())
        while (value <= 0.0) and (value > 1.0):
            print(error_message)
            value = float(input())
        return value
    if type == 'string':
        value = input()
        while value not in string_range:
            print(error_message)
            value = input()
        return value

if __name__=='__main__':
    main()
