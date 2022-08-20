import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Task (1)
# Creating the class of Perceptron, the input is the number of iteration and learning rate
class perceptron(object):
    def __init__(self, number_iteration, learning_rate):
        self.number_iteration = number_iteration
        self.learning_rate = learning_rate
    # Calculating the output of the neural network for the desired input with obtained weights from the train part
    def prediction(self, input):
        # calculating the net input
        z = np.dot(input, self.w) + self.w0
        # Applying the threshold function
        y_hat = 1 if z >= 0 else -1
        return y_hat
    # Training the neural network with labeled data
    def train(self, input, label):
        # Initializing the wight vector
        self.w = np.random.rand(input.shape[1])
        self.w0 = np.random.rand()
        self.cost = []
        self.accuracy = []
        # Training the network for the desired number of iterations
        for i in range(self.number_iteration):
            acc = 0
            error1 = 0
            # Choose the pair of input and label in each iteration
            for x, y in zip(input, label):
                # Predicting the output of network for the input
                y_hat = self.prediction(x)
                # Calculating the error between the label and predicted value for each input
                error = (y - y_hat) * self.learning_rate
                # Updating the values of w0 and w
                delta_w0 = error
                #delta_w = np.multiply(error, x)
                delta_w = error * x
                self.w0 = self.w0 + delta_w0
                self.w = self.w + delta_w
                error1 = error1 + int(error != 0.0)
                acc += 1 if error == 0 else 0
            # Calculating the accuracy of network in each iteration
            self.accuracy.append((acc/input.shape[0])*100)
            # Calculating the cost of network in each iteration
            self.cost.append(error1)
        return self.accuracy, self.cost
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Task (2)
# Creating the class of Adaline, the input is the number of iteration and learning rate
class adaline(object):
    def __init__(self, number_iteration, learning_rate):
        self.number_iteration=number_iteration
        self.learning_rate=learning_rate
    # calculating the output of activation function in adaline structure
    def activation(self, input):
        y_hat = np.dot(input, self.w) + self.w0
        return y_hat
    # Calculating the output of the neural network for the desired input with obtained weights from the train part
    def prediction(self, input):
        y_hat1 = self.activation(input)
        y_hat = np.where(y_hat1 >= 0, 1, -1)
        return y_hat
    # Training the neural network with labeled data
    def train(self, input, label):
        # Initializing the wight vector
        self.w = np.random.rand(input.shape[1])
        self.w0 = np.random.rand()
        self.cost = []
        self.accuracy = []
        acc = 0
        # Training the network for the desired number of iterations
        for j in range(self.number_iteration):
            # Predicting the output of activation function for the whole training dataset
            y_hat = self.activation(input)
            # Calculating the error between the label and predicted value for the whole training dataset
            error = label - y_hat
            # Calculating the gradient based on the whole training dataset and updating the weight vector
            delta_w0 = error.sum() * self.learning_rate
            delta_w = input.T.dot(error) * self.learning_rate
            self.w0 = self.w0 + delta_w0
            self.w = self.w + delta_w
            # Collecting the cost values to check the convergence
            cost_i = (error ** 2).sum()/2
            self.cost.append(cost_i)
            # Calculating the accuracy of network in each iteration
            y_hat1 = self.prediction(input)
            error1 = label - y_hat1
            acc = np.where(error1 == 0, 1, 0)
            self.accuracy.append((acc.sum()/input.shape[0])*100)
        return self.accuracy, self.cost
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Task (3)
# Creating the class of Stochastic Gradient Descent (SGD), the input is the number of iteration and learning rate
class sgd(object):
    def __init__(self, number_iteration, learning_rate):
        self.number_iteration = number_iteration
        self.learning_rate = learning_rate
    # calculating the output of activation function in SGD structure
    def activation(self, input):
            y_hat = np.dot(input, self.w) + self.w0
            return y_hat
    # Calculating the output of the neural network for the desired input with obtained weights from the train part
    def prediction(self, input):
            y_hat1 = self.activation(input)
            y_hat = 1 if y_hat1 >= 0 else -1
            return y_hat
    # Training the neural network with labeled data
    def train(self, input, label):
        # Initializing the wight vector
        self.w = np.random.rand(input.shape[1])
        self.w0 = np.random.rand()
        self.cost = []
        self.accuracy = []
        # Training the network for the desired number of iterations
        for i in range(self.number_iteration):
            # Shuffling the data
            index = np.random.permutation(len(label))
            input = input[index]
            label = label[index]
            cost = []
            acc = 0
            # Choose the pair of input and label in each iteration
            for x, y in zip(input, label):
                # Predicting the output of activation function for the input
                y_hat = self.activation(x)
                # Calculating the error between the label and predicted value for each input
                error = y - y_hat
                # Updating the values of w0 and w
                self.w = self.w + (self.learning_rate * x.dot(error))
                self.w0 = self.w0 + (self.learning_rate * error)
                # Collecting the cost values
                cost1 = 0.5 * (error ** 2)
                cost.append(cost1)
                # Calculating the accuracy of network in each iteration
                y_hat1 = self.prediction(x)
                error1 = y - y_hat1
                acc += 1 if error1 == 0 else 0
            self.accuracy.append((acc / input.shape[0]) * 100)
            # Calculating the cost averaged in each iteration to check the convergence
            mean_cost = sum(cost) / len(label)
            self.cost.append(mean_cost)
        return self.accuracy, self.cost
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Task (4)
# Creating a class to test different classifier for the desired data
class classifier (object):
     # The input is type of binary classifier (perceptron, adaline, sgd) and the url of the desired data
    def __init__(self, type_classifier, data_path):
        # If the name of classifier is not valid, it shows error
        if type(type_classifier) in [type(perceptron), type(adaline), type(sgd)]:
            self.type_classifier = type_classifier
            self.data_path = data_path
            # Read data from url
            data = pd.read_csv(self.data_path, header=None)
            # Preparing data for binary classification
            self.input = data.iloc[:, 0:-1].values
            self.label = data.iloc[:, -1].values
            # If there is more than two class in the data, it prepares the data for binary classification
            # It considers the first class as the binary output of 1 and the other as -1
            y = data.iloc[:, -1].unique()
            self.label = np.where(self.label == y[0], 1, -1)
    # Training the created object based on the entered type of classifier
    def classifier_train(self, number_iteration, leraning_rate):
        self.network = self.type_classifier(number_iteration, leraning_rate)
        self.cost = self.network.train(self.input, self.label)
        return self.cost
    # Predict the desired input based on the trained object
    # The input is the number of row of data, to can get input and its label to calculate the prediction error
    def classifier_prediction(self, number):
        self.prediction = self.network.prediction(self.input[number,:])
        self.error = self.label[number] - self.prediction
        return self.prediction, self.error
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# The url of datasets
'''RIS dataset'''
#url = "https://www.cs.nmsu.edu/~hcao/teaching/cs487519/data/iris.data"
'''SONAR dataset'''
#url='https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Call the classifier class with different binary classification for the desired data

'''Perceptron:
     For IRIS dataset, learning rate is 0.01 nd the number of iteration is 10
     For SONAR dataset, learning rate is 0.005 nd the number of iteration is 40
   Adaline:
     For IRIS dataset, learning rate is 0.0001 nd the number of iteration is 10
     For SONAR dataset, learning rate is 0.001 nd the number of iteration is 50
   SGD:
     For IRIS dataset, learning rate is 0.005 nd the number of iteration is 5
     For SONAR dataset, learning rate is 0.005 nd the number of iteration is 50'''

#type_classifier = perceptron
#number_iteration = 10
#learning_rate = 0.01

#start_time = time.time()
#model=classifier(type_classifier, url)
#[accuracy, cost]=model.classifier_train(number_iteration, learning_rate)
#print(" %s seconds " % (time.time() - start_time))

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Task (5)
# Plot the cost of classifier in each iteration
#plt.plot(range(len(cost)), cost)
#plt.xlabel("Number of iteration")
#plt.ylabel("Cost")
#plt.title("%s" %type_classifier)
#plt.show()

# Show the accuracy of classifier algorithms in each iteration
#print(accuracy)

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Task (7)
# Implementing the multiclass classifier using One-vs-Rest strategy with desired classifier
class one_vs_rest (object):
    # The input is type of classifier, url of data, the number of iteration and the learning rate
    def __init__(self, type_classifier, data_path, number_iteration, learning_rate):
        self.type_classifier = type_classifier
        self.number_iteration = number_iteration
        self.learning_rate = learning_rate
        self.data_path = data_path
        # Read the data from url
        data = pd.read_csv(self.data_path, header=None)
        # Prepare data for One-vs-Rest strategy
        self.input = data.iloc[:, 0:-1].values
        self.label = data.iloc[:, -1].values
        self.y = np.unique(self.label)
        self.c = []
        # Train the network for the class denoted for 1 while the rest of classes denoted by -1
        for i in range(len(self.y)):
            self.label1 = np.where(self.label == self.y[i], 1, -1)
            self.network = self.type_classifier(number_iteration, learning_rate)
            self.network.train(self.input, self.label1)
            self.c.append(self.network)
    # Predict the class of the desired input based on the One-vs-Rest algorithm
    # The input is allocated to the class that the output is 1
    def prediction(self, x):
        for i in range(len(self.y)):
            self.prediction = self.c[i].prediction(x)
            j = i+1
            print("class_prediction.%s=" %j, self.prediction)
        return self.prediction
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Call the One-vs-Rest class for the desired data
'''IRIS'''
url = 'https://www.cs.nmsu.edu/~hcao/teaching/cs487519/data/iris.data'
# Training the One-vs-Rest strategy with the desired binary classification
model=one_vs_rest(sgd, url, 10, 0.001)
#Predict for desired input
p=model.prediction([4., 3.2, 1.3, 0.2])

'''Glass'''
#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
# Training the One-vs-Rest strategy with the desired binary classification
model=one_vs_rest(sgd, url, 30, 0.00001)
#Predict for desired input
p=model.prediction([2, 1.51761, 13.89, 3.6, 1.36, 72.73, 0.48, 7.83, 0., 0.])

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
