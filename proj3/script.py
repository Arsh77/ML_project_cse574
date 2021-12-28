import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from math import sqrt


def preprocess():
    """ 
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))
    initialWeights.resize((n_features+1, 1))

    train_data = np.concatenate((np.ones((n_data, 1)), train_data), axis=1)
    theta = sigmoid(train_data @ initialWeights)

    error = (-1/n_data) * np.sum(labeli * np.log(theta) + (1 - labeli) * np.log(1 - theta))

    error_grad = (1/n_data) * np.sum((theta - labeli) * train_data, axis=0)
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    
    data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)
    posterior = data @ W
    label = np.argmax(posterior, axis=1)
    label.resize((data.shape[0], 1))
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 10 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    train_data = np.concatenate((np.ones((n_data, 1)), train_data), axis=1)
    params.resize((n_feature + 1, n_class))

    prob_num = np.exp(train_data @ params)

    row_sum = np.sum(prob_num, axis=1)
    row_sum.resize((n_data, 1))

    theta = prob_num / row_sum

    error = -(1/n_data)*(np.sum(np.sum(labeli * np.log(theta))))

    error_grad = (1/n_data)*(train_data.T @ (theta - labeli)).flatten()

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data


    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)
    prediction = np.exp(data @ W)

    pred_label = np.argmax(prediction, axis = 1)

    label = pred_label.reshape((data.shape[0], 1))
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

# number of validation samples
n_validation = validation_data.shape[0]

# number of test samples
n_test = test_data.shape[0]

Y = np.zeros((n_train, n_class))
X = np.zeros((n_validation, n_class))
Z = np.zeros((n_test, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()
    X[:, i] = (validation_label == i).astype(int).ravel()
    Z[:, i] = (test_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}

print('Training of One vs All Logistic Regression model started.')

for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    print('Training for digit', i, 'complete.')
    W[:, i] = nn_params.x.reshape((n_feature + 1,))
    #Code for gathering data to write in the report.
    '''
    weight = W[:, i].copy()
    error, temp = blrObjFunction(weight, train_data, labeli)
    print('For Digit ', i, 'Training error = ', error)
    labeli = X[:, i].reshape(n_validation, 1)
    error, temp = blrObjFunction(weight, validation_data, labeli)
    print('For Digit ', i, 'Validation error = ', error)
    labeli = Z[:, i].reshape(n_test, 1)
    error, temp = blrObjFunction(weight, test_data, labeli)
    print('For Digit ', i, 'Test error = ', error)
    '''

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################


#creating random sample
index1 = np.random.choice(train_data.shape[0], int(0.2*train_data.shape[0]), replace=False)  
X_train_data = train_data[index1]
y_train_data = train_label[index1]
index2 = np.random.choice(validation_data.shape[0], int(0.2*validation_data.shape[0]), replace=False)  
X_validation_Data = validation_data[index2]
y_validation_Data = validation_label[index2]
index3 = np.random.choice(test_data.shape[0], int(0.2*test_data.shape[0]), replace=False)  
X_test_data = test_data[index3]
y_test_data = test_label[index3]


# SVM with linear 
clf = SVC(kernel='linear')

# Training SVM with linear kernal
clf.fit(X_train_data, y_train_data.flatten()) 

#SVM prediction on training data
linearPredictionYTrainingData = clf.predict(X_train_data)
accTrainingData = accuracy_score(linearPredictionYTrainingData, y_train_data)*100
print('\n Accuracy of SVM using linear kernel on training data: ' + str(accTrainingData) + '%\n')

#SVM prediction on validation data
linearPredictionYValidationData = clf.predict(X_validation_Data)
accValidationData = accuracy_score(linearPredictionYValidationData, y_validation_Data)*100
print('\n Accuracy of SVM using linear kernel on validation data: ' + str(accValidationData) + '%\n')

#SVM prediction on test data
linearPredictionYTestData = clf.predict(X_test_data)
accTestData = accuracy_score(linearPredictionYTestData, y_test_data)*100
print('\n Accuracy of SVM using linear kernel on test data: ' + str(accTestData) + '%\n')


# SVM with rbf with gamma=1
clf = SVC(kernel='rbf', gamma=1)
# Training SVM with linear kernal
clf.fit(X_train_data, y_train_data.flatten()) 

#SVM prediction on training data
yPredictionTrainingData = clf.predict(X_train_data)
accTrainingData = accuracy_score(yPredictionTrainingData, y_train_data)*100
print('\n Accuracy of SVM using rbf kernel with gamma = 1 on training data: ' + str(accTrainingData) + '%\n')

#SVM prediction on validation data
yPredictionValidationData = clf.predict(X_validation_Data)
accValidationData = accuracy_score(yPredictionValidationData, y_validation_Data)*100
print('\n Accuracy of SVM using rbf kernel with gamma = 1 on validation data: ' + str(accValidationData) + '%\n')

#SVM prediction on test data
yPredictionTestData = clf.predict(X_test_data)
accTestData = accuracy_score(yPredictionTestData, y_test_data)*100
print('\n Accuracy of SVM using rbf kernel with gamma = 1 on test data: ' + str(accTestData) + '%\n')


#SVM with rbf with gamma with default value [gamma default value is 'scale']
#By not adding gamma parameter in SVC() gamma value used by function are default values
clf = SVC(kernel='rbf')
# Training SVM with linear kernal
clf.fit(X_train_data, y_train_data.flatten()) 

#SVM prediction on training data
yPredictionTrainingData = clf.predict(X_train_data)
accTrainingData = accuracy_score(yPredictionTrainingData, y_train_data)*100
print('\n Accuracy of SVM using rbf kernel with gamma = scale on training data: ' + str(accTrainingData) + '%\n')

#SVM prediction on validation data
yPredictionValidationData = clf.predict(X_validation_Data)
accValidationData = accuracy_score(yPredictionValidationData, y_validation_Data)*100
print('\n Accuracy of SVM using rbf kernel with gamma = scale on validation data: ' + str(accValidationData) + '%\n')

#SVM prediction on test data
yPredictionTestData = clf.predict(X_test_data)
accTestData = accuracy_score(yPredictionTestData, y_test_data)*100
print('\n Accuracy of SVM using rbf kernel with gamma = scale kernel on test data: ' + str(accTestData) + '%\n')

#C values
C = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
yAcc1 = []
yAcc2 = []
yAcc3 = [] 

for i in C:    
    # SVM with rbf with gamma with default value [gamma default value is 'scale'] and C values belonging to [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 
    clf = SVC(kernel='rbf', C=i)
    # Training SVM with linear kernal
    clf.fit(X_train_data, y_train_data.flatten()) 

    #SVM prediction on training data
    yPredictionTrainingData = clf.predict(X_train_data)
    accTrainingData = accuracy_score(yPredictionTrainingData, y_train_data)*100
    yAcc1.append(accTrainingData)
    print('\n Accuracy of SVM using rbf kernel with gamma = scale kernel and value of C = ', i ,'on training data: ' + str(accTrainingData) + '%\n')

    #SVM prediction on validation data
    yPredictionValidationData = clf.predict(X_validation_Data)
    accValidationData = accuracy_score(yPredictionValidationData, y_validation_Data)*100
    yAcc2.append(accValidationData)
    print('\n Accuracy of SVM using rbf kernel with gamma = scale kernel and value of C = ', i ,'on validation data: ' + str(accValidationData) + '%\n')

    #SVM prediction on test data
    yPredictionTestData = clf.predict(X_test_data)
    accTestData = accuracy_score(yPredictionTestData, y_test_data)*100
    yAcc3.append(accTestData)
    print('\n Accuracy of SVM using rbf kernel with gamma = scale kernel and value of C = ', i ,'on test data: ' + str(accTestData) + '%\n')


#Training full data on rbf kernel, default value of gamma and C = 20
clf = SVC(kernel='rbf', C = 20)
#Validation on classifer trained over full data on rbf kernel, default value of gamma and C = 20
clf.fit(train_data, train_label.flatten()) 

#Testing on classifer trained over full data on rbf kernel, default value of gamma and C = 20
yPredictionTrainingData = clf.predict(train_data)
accTrainingData = accuracy_score(yPredictionTrainingData, train_label.flatten())*100
print('\n Accuracy of SVM using rbf kernel with gamma = scale on training data: ' + str(accTrainingData) + '%\n')

#SVM prediction on validation data
yPredictionValidationData = clf.predict(validation_data)
accValidationData = accuracy_score(yPredictionValidationData, validation_label)*100
print('\n Accuracy of SVM using rbf kernel with gamma = scale on validation data: ' + str(accValidationData) + '%\n')

#SVM prediction on test data
yPredictionTestData = clf.predict(test_data)
accTestData = accuracy_score(yPredictionTestData, test_label)*100
print('\n Accuracy of SVM using rbf kernel with gamma = scale kernel on test data: ' + str(accTestData) + '%\n')

plt.figure(figsize=(12,8))
plt.plot(C,yAcc1, marker = "s",ms=12,mec='k', linewidth=8, label = "training data")
plt.plot(C,yAcc2, marker = "s", ms=12,mec='k', linewidth=8, label = "validation data")
plt.plot(C,yAcc3, marker = "s",ms=12,mec='k', linewidth=8, label = "test data")
plt.title('Accuracy vs C')
plt.xlabel('C values')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""
Script for Extra Credit Part
"""
print('\nTraining for Multiclass Logistic Regression model started.')
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

#Uncomment the below section to print error for each dataset.
'''
weight = W_b.copy()
error, temp = mlrObjFunction(weight, train_data, Y)
print('train error = ', error)
error, temp = mlrObjFunction(weight, validation_data, X)
print('validation error = ', error)
error, temp = mlrObjFunction(weight, test_data, Z)
print('test error = ', error)
'''

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
