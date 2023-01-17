"""
Finally I can write the .py files in my desktop, regardless I will be adding my notes on this later on, this is simply a solution to one of 
Dr. Andrew NG problems on the Deep Learning specialization course, although this problem in theory does reflect how a CNN will look like but 
this is simply a shallow neural network with one hidden layer, regardless I will add my comments on this later on
"""


import numpy as np
import copy
import matplotlib.pyplot as plt
from testCases_v2 import *
from public_tests import *
import sklearn
import sklearn.datasets
import sklearn.linear_model;

def layer_sizes(X, Y):  
    n_x=X.shape[0];
    n_h=4;
    n_y=Y.shape[0]
    
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
 

 
    #n_x tells us the number of features and number of weights 
    #n_h tells us the number of nuerons in the layer
    W1=np.random.randn(n_h,n_x)*0.01;
    b1=np.zeros((n_h,1));
    W2=np.random.randn(n_y,n_h)*0.01;
    b2=np.zeros((n_y,1));
    

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):

    

    Z1=np.dot(parameters["W1"],X)+parameters["b1"];
    A1=np.tanh(Z1);
    Z2=np.dot(parameters["W2"],A1)+parameters["b2"];
    A2=sigmoid(Z2);
   
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A, Y):

    m = Y.shape[1] # number of examples

    
    """
    np.dot(Y,np.log(A)) + (1-Y)*(np.log(1-A)   )
    
    
    """
    cost=0;

    cost=        (   (Y)*np.log(A)+ (1-Y)*np.log(1-A)   )    
    cost=-cost/m;
    cost=cost.sum(axis=1,keepdims=True);
    
    
    cost = float(np.squeeze(cost))  # Please if you read this and know why this solves a dimensions error please tell me.
                                    

    return cost

def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]
    
    
    W1=parameters["W1"];
    W2=parameters["W2"];
        
    A1=cache["A1"];
    A2=cache["A2"];
 
    
    dZ2 =(A2-Y);#how much was y-hat off correct prediction
    dW2 =np.dot(dZ2,A1.T)/m;#how much off y-hat times the input values(aka the output values of Layer 1) all divided by number of examples
    db2 = dZ2.sum(axis=1,keepdims=True)/m;#sum all off predictions
    dZ1 = W2.T*dZ2*(1 - np.power(A1, 2));#
    dW1 = 1/m*(np.dot(dZ1,X.T));
    db1 = 1/m*dZ1.sum(axis=1,keepdims=True);
    

    

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):

    W1=parameters["W1"]-learning_rate*grads["dW1"];
    b1=parameters["b1"]-learning_rate*grads["db1"];
    W2=parameters["W2"]-learning_rate*grads["dW2"];
    b2=parameters["b2"]-learning_rate*grads["db2"];
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):

    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    
    #W1=np.random.randn(n_x,n_h)*0.01;
    #b1=np.zeros((n_x,1));
    #W2=np.random.randn(1,n_x)*0.01;
    #b2=0;
    
    parameters=initialize_parameters(n_x, n_h, n_y);

    for i in range(0, num_iterations):
         

      
        
        A2, cache = forward_propagation(X, parameters);
        cost = compute_cost(A2, t_Y);
            
        grads = backward_propagation(parameters, cache, X, Y);
        
       
        parameters = update_parameters(parameters, grads, learning_rate = 1.2); 
     
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


def predict(parameters, X):


    A2, cache = forward_propagation(X, parameters);
    predictions=np.zeros(A2.shape[0]);
    
    
    predictions = (A2 > 0.5);

    
    return predictions
