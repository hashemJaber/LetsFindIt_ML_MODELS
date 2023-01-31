import numpy as np;
def initialize_parameters(n_x, n_h, n_y):
    
    np.random.seed(1)
    
 
    W1=np.random.randn(n_h,n_x)*0.01;
    b1=np.zeros((n_h,1));
    W2=np.random.randn(n_y,n_h)*0.01;
    b2=np.zeros((n_y,1));
    
    # YOUR CODE ENDS HERE
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
            
def initialize_parameters_deep(layer_dims):

    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network
    print(str(layer_dims))
    for l in range(1, L):
        
        parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01;
        parameters["b"+str(l)]=np.zeros((layer_dims[l],1));
  
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters
def linear_forward(A, W, b):

  
    Z=np.dot(W,A)+b
    
    
    cache = (A, W, b)
    
    return Z, cache
    
    
def linear_activation_forward(A_prev, W, b, activation):

    A=3;
    
    if activation == "sigmoid":
        
        
        
        Z,linear_cache=linear_forward(A_prev, W, b);
        A, activation_cache = sigmoid(Z)
        
        
    
    elif activation == "relu":
        
        Z,linear_cache=linear_forward(A_prev, W, b);
        A, activation_cache = relu(Z)
        
        # YOUR CODE ENDS HERE
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    

    caches = []
    A = X
    L = len(parameters) // 2
    A_prev = A
    for l in range(1, L):
        
        A_prev = A
       
        A,cache=linear_activation_forward(A_prev,parameters['W' + str(l)] , parameters['b' + str(l)], "relu");
        caches.append(cache);
        

    
        
    A_prev = A
        
    AL,cache=linear_activation_forward(A_prev,parameters['W' + str(L)] , parameters['b'+str(L)], "sigmoid");
    caches.append(cache);
    

          
    return AL, caches
    
def compute_cost(AL, Y):

    
    m = Y.shape[1]

    
    cost=   ( (Y)*np.log(AL)+ (1-Y)*np.log(1-AL)    )
    cost=cost.sum(axis=1,keepdims=True);
    cost=-cost/Y.shape[1];
    

    
    cost = np.squeeze(cost)

    
    return cost
    
def linear_backward(dZ, cache):
   
    A_prev, W, b = cache
    m = A_prev.shape[1]

    
    dW =           1 / m * (np.dot(dZ,A_prev.T))
    db = 1 / m * (np.sum(dZ,axis = 1,keepdims = True))
    dA_prev = np.dot(W.T,dZ)
    

    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
        
        
        # YOUR CODE ENDS HERE
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
        
    
    return dA_prev, dW, db
    
def L_model_backward(AL, Y, caches):
   
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    

    
  
        
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    
    
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
   
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        

    return grads

def update_parameters(params, grads, learning_rate):
    
    parameters = params.copy()
    L = len(parameters) // 2 # number of layers in the neural network

  
    for l in range(L):
     
        parameters["W" + str(l+1)]=parameters["W" + str(l+1)]-learning_rate*(grads["W"+str(l+1)])
        parameters["b" + str(l+1)] =parameters["b" + str(l+1)]-learning_rate*(grads["b"+str(l+1)])
        
        

    return parameters
    
    
def derive_relu(Z):
   return Z>0;
   

def derive_sigmoid(A,Y):
    return A-Y;



print("I will creat a class from all of these functions and strap them together, building something like my own library or class");

class DNN:

  def __init__(self, learning_rate, Lambda,X,Y,layer_dims,epochs=2500,activations=
  
  np.array(["relu","relu","relu","sigmoid"])   ):
    self.learning_rate = learning_rate
    self.Lambda = Lambda
    self.X=X
    self.Y=Y
    self.layer_dims=layer_dims
    self.epochs=epochs
    self.activations=activations
    #assert(layer_dims.shape[0]==activations.shape[0])
    

dnn_1 = DNN(0.001,0.0001,X=np.array([
[1,2,3,4],
[5,6,7,8],
[9,10,11,12]
]),
Y=np.array([1,1,0]

),

layer_dims=np.array([2,3,4]   ))
    


print(str(dnn_1.activations));
#to do : add a regulria=zation method
#to do : add a dropout tool for every layer
#to do: add a Adam optimization and minibatching hyper parameters
