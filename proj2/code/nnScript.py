import numpy as np
import pickle
import pandas as pd
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
from datetime import datetime


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1/(1 + np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    #Assuming only the 100 pixels in the middle of the image are useful. So we will take out 10 x 10 window from the middle.
    data = mat['train0']
    label = np.zeros((mat['train0'].shape[0], 1))
    test_data = mat['test0']
    test_label = np.zeros((mat['test0'].shape[0], 1))
    for i in range(1, 10):
      data = np.concatenate((data, mat['train'+str(i)]))
      label = np.concatenate((label, i*np.ones((mat['train'+str(i)].shape[0], 1))))
      test_data = np.concatenate((test_data, mat['test'+str(i)][:, :]))
      test_label = np.concatenate((test_label, i*np.ones((mat['test'+str(i)].shape[0], 1))))

    #Finding index of columns where all values are same
    idx = np.argwhere(np.all(data[..., :] == 0, axis=0))

    #Removing zero value columns from all three datasets
    data = np.delete(data, idx, axis=1)
    test_data = np.delete(test_data, idx, axis=1)

    #Normalizing the data by dividing all values by 255.
    data = data/255
    test_data = test_data/255

    data_permutation = np.random.permutation(len(data))
    data = data[data_permutation]
    label = label[data_permutation]

    test_permutation = np.random.permutation(len(test_data))
    test_data = test_data[test_permutation]
    test_label = test_label[test_permutation]

    train_data = data[0:50000, :]
    train_label = label[0:50000, :]

    validation_data = data[50000:, :]
    validation_label = label[50000:, :]
    # Feature selection
    # Your code here.

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    

    # Your code here
    #
    #
    #
    #
    #
    training_data = np.concatenate((np.ones((training_data.shape[0], 1)), training_data), 1)
    a1 = training_data @ w1.T
    z1 = sigmoid(a1)
    z1 = np.concatenate((np.ones((a1.shape[0], 1)), z1), 1)
    a2 = z1 @ w2.T
    o = sigmoid(a2)
    training_label.resize((training_label.shape[0],))
    one_hot_labels = np.eye(n_class)[training_label.astype(int)]
    
    obj_val = (-1/training_data.shape[0]) * np.sum(one_hot_labels * np.log(o) + (1 - one_hot_labels)*np.log(1 - o))
    obj_val = obj_val + (lambdaval/(2*training_data.shape[0])) * (np.trace(w1 @ w1.T) + np.trace(w2 @ w2.T))

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    grad_w2 = (1/training_data.shape[0])*((o - one_hot_labels).T @ z1 + lambdaval * w2)
    grad_w1 = (1/training_data.shape[0])*(((((1 - z1[:, 1:]) * z1[:, 1:]) * ((o - one_hot_labels) @ w2[:, 1:])).T @ training_data) + lambdaval * w1)
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % labels: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    data = np.concatenate((np.ones((data.shape[0], 1)), data), 1)
    z1 = data @ w1.T
    a1 = sigmoid(z1)
    a1 = np.concatenate((np.ones((a1.shape[0], 1)), a1), 1)
    z2 = a1 @ w2.T
    o = sigmoid(z2)
    labels = np.argmax(o, axis = 1)
    #Remove the next line if labels are needed in (n,) dimension.
    labels.resize((len(labels), 1))
    return labels


"""**************Neural Network Script Starts here********************************"""

result_dict = {'lambda':[], 'hidden_units':[], 'train_acc':[], 'val_acc':[], 'test_acc':[], 'train_time':[]}

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
for n_hidden in range(4, 21, 4):
# set the number of nodes in output unit
  n_class = 10

  # initialize the weights into some random matrices
  initial_w1 = initializeWeights(n_input, n_hidden)
  initial_w2 = initializeWeights(n_hidden, n_class)

  # unroll 2 weight matrices into single column vector
  initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

  # set the regularization hyper-parameter
  training_accuracy = []
  validation_accuracy = []
  test_accuracy = []
  lambdas = []
  for lambdaval in range(0, 61, 10):
    result_dict['lambda'].append(lambdaval)
    result_dict['hidden_units'].append(n_hidden)
    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    start = datetime.now()

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    result_dict['train_time'].append((datetime.now() - start).total_seconds())

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    train_accuracy = 100 * np.mean((predicted_label.reshape((predicted_label.shape[0],)) == train_label.reshape((train_label.shape[0],))).astype(float))
    training_accuracy.append(train_accuracy)
    result_dict['train_acc'].append(train_accuracy)

    print('\n Training set Accuracy:' + str(train_accuracy) + '%')

    lambdas.append(lambdaval)
    
    predicted_label = nnPredict(w1, w2, validation_data)

    val_accuracy = 100 * np.mean((predicted_label == validation_label).astype(float))

    validation_accuracy.append(val_accuracy)
    result_dict['val_acc'].append(val_accuracy)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(val_accuracy) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Test Dataset

    tes_accuracy = 100 * np.mean((predicted_label == test_label).astype(float))
    test_accuracy.append(tes_accuracy)
    result_dict['test_acc'].append(tes_accuracy)

    print('\n Test set Accuracy:' + str(tes_accuracy) + '%')

  plt.plot(lambdas, training_accuracy, 'ro-', label='Training Data')
  plt.plot(lambdas, validation_accuracy, 'bo-', label='Validation Data')
  plt.plot(lambdas, test_accuracy, 'ko-', label='Test Data')
  plt.legend(loc="upper right")
  plt.xlabel('Regularization parameter (lambda)')
  plt.ylabel('Accuracy')
  plt.title('Number of hidden units = '+str(n_hidden))
  plt.show()
