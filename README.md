# LetsFindIt_ML_MODELS
This is all the ML models I will be using in my application, as soon as I have tested a model and get to a good performance level I will publish it here
#THIS IS THE LOGISTIC MODEL PROTOTYPE I WILL BE USING FOR LetsFindIt
#image recognition model, it will be later on  updated to a softmax 
#algorithm for my multi-classification model
#NOTE:DATA WONT BE PROVIDED OR BE VISIBLE AS THIS IS  
#A PROTOTYPE MODEL 

#Needed Dependecny
import numpy as np
#FOR DEEP COPY
import copy
#FOR PLOTTING IMAGES, NOT USED  HERE, JUST A LEAD IF YOU WISH TO SEE INTRESTING
# DATA SUCH AS COST CHANGE AFTER %NUMBER OF ITERATIONS OR AFTER ADJUSTING THE LEARNING RATE
import matplotlib.pyplot as plt


#FIRST STEP FLATTEN YOUR IMAGE 
#CONSIDERING WE ARE USING A NEURAL NETWORK MINDSET WITH ONE ACTIVATION 
#NUERON AS A LAYER THIS IS NEEDED, ORDER DOES NOT MATTER AS LONG AS YOU
#STAY CONSISTENT WITH THE RESHAPING OF IMAGES
#THE FOLLOWING IS SIMPLY ONE WAY TO RESHAPE YOUR IMAGE UTILIZING 
#VECTORIZATION TO SPEED UP THE PROCESS
#NOTE: T IS FOR TRANSPOSING
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T ;
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T; 


#SECOND STEP, NORMALIZE YOUR DATA TO SPEED UP THE PROCESS IN LESS 
#ITERATIONS/LEARNING RATE THAN NEEEDED
#THERE ARE MANY NORMALIZATION FORMULAS BUT THE FOLLOWING IS WHAT I 
#WAS ADVISED TO USE
#NOTE: ADD THE . AT THE END OR ELSE PRECISION MAY BE LOST
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.
#OTHER NORMALIZATION FORMULAS ARE 
#1)RESCALLING 
#X=(X-MIN(X))/(MAX(X)-MIN(X))
#MIN MEANS SMALLEST NUMBER
#MAX MEANS LARGEST NUMBER

#NOTE:USE () WHEN ATTEMPTING ADDITION/SUBTRACTION OR ELSE DIVIDE AND MULTIPLICATION WILL TAKE
#HIGHER PRECEDENCE

#2)MEAN NORMALIZATION //HIGHLY USED BY ME
#X=(X-AVERAGE(X))/(MAX(X)-MIN(X) )
#MIN MEANS SMALLEST NUMBER
#MAX MEANS LARGEST NUMBER
#AVERAGE MEANS AVERAGE OR THE MEAN OF ALL VALUES

#FOR OTHER NORMALIZATION FORMULAS VISIT 
#https://learn.microsoft.com/en-us/azure/machine-learning/component-reference/normalize-data
#DONT WORRY THEY PROVIDE THE MATH FOR IT AS WELL  AND NOT LIMITED TO
# Azure Machine Learning designer

#THIRD: THE SIGMOID FUNCTION aka activation function:
#THIS IS WHERE OUR HYPOTHESIS Z WILL GO TO SEE THE PERCENTAGE 
#OF HOW LIKELY IS THE IMAGE SIMILAR TO OUR SUPERVISED DESIRED OUTPUT
#NOTE: STICK TO NP.EXP() FOR BETTER PERFORMANCE OR ELSE USE e NORMALLY
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s



#THE FOLLOWING FUNCTION IS TAKEN FROM ANDREW NG FOR INITIALIZING VALUES
#OF WEIGHTS AND BIAS, YOU CAN DO IT YOURSELF BUT THIS 
#WILL REDUCE HOURS OF FRUSTRATING DIMENSION ERRORS FOR YOUR 
#VECTORS FOR THE WEIGHTS, BELIEVE ME, AND TAKE MY WORD ON IT
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    """
    
    # (≈ 2 lines of code)
    # w = ...
    # b = ...

    w=np.zeros((dim,1));
    b=0.0;
    print(w);
    


    return w, b

#FOURTH STEP: IMPLEMENTING FORWARD PROPAGATION AND BACKWARD PROPAGATION 

def propagate(w, b, X, Y):
    #m FOR NUMBER OF EXAMPLES 
    m = X.shape[1];
    #n FOR  NUMBER OF FEATURES
    n = X.shape[0];

    cost=0;
    
    A=np.zeros([n,1]);
   # print(A);
   # print(X);
   # print(w);
   # print(Y);
   
    #A if for PROBABILITY OF EXAMPLE i BEING 0(FALSE) OR 1(TRUE) OF BEING 
    #THE DESIRED OUTPUT    


    A=sigmoid(np.dot(w.T,X)+b); 
    #YES THE LOSS IS IN NEGATIVE, NO WE DONT USE MEAN SQUEARD ERROR WE USE LOG,
    #HENCE THE NAME THE LOGESTICE REGRESSION MODEL, IF YOU ARE THINKING ABOUT MSE THEN YOU MOST LIKELY ARE LOOKING FOR 
    #A LINEAR REGRESSION MODEL
    cost = -1/m * (np.dot(Y,np.log(A).T) + np.dot((1-Y),np.log(1 - A).T)) 

    #print("X is what ?, its this : "+str(X));  
    #print("w is what ?, its this : "+str(w)); 
    #dw means derivative of cost/loss with respect of w
    dw=(1/m)*(np.dot(X,(A - Y).T));
    #db means derivative of cost/loss with respect of b
    db=(1/m)*np.sum(A-Y);
    #USE THIS, I HAVE SOMETHING WRONG WITH MY CODE WHICH THIS HELPS TO 
    #FIX FOR SOME ODD REASON
    cost = np.squeeze(np.array(cost))
    
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost



#FINALLY GRADIENT DESCENT ALGORITHM
  def optimize(w, b, X, Y, num_iterations=2000, learning_rate=0.001, print_cost=False):
  
    
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    #RECORD YOUR PERFORMACE, ITS IMERATIVE WHEN AUDITING THE MODEL 
    costs = []
    
    for i in range(num_iterations):
     
        grads, cost=propagate(w,b,X,Y);
        dw = grads["dw"]
        db = grads["db"]
        
      #THIS IS THE HOLLY MATH FOR UPDATING WEIGHTS AND BIAS
      #dw is the derivaitve with respect to w NOTE:w is not a scalar 
      #DO NOT GET TRICKED! WE ARE UTILIZING BROADCASTING AND PARALLEL PROCESSING 
      #TO SPEED THINGS UP
      #db is the derivative with respect to b 
        w=w-learning_rate*dw;
        b=b-learning_rate*db;
       
        
        # Record the costs for every 200 iterations
        #NOTE: yes in every iteration we go through the full training 
        #set, which is why we are USING VECTORIZED IMPLEMENTATION 
        #TO SPEED THINGS UP, IF YOU HAVE A WAY TOO LARGE OF A DATA SET
        #THEN IMPLEMENT MINI BATCHING WITH SOFT LEARNING.

        if i % 200 == 0:
            costs.append(cost)
        
            # Print the cost every 2OO training iterations, YOU CAN PLAY 
            #WITH IT
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs  

 def predict(w, b, X):
    
    Y_prediction = np.zeros((1, X.shape[1]))
    #USE THE FUNCTION THAT ANDREW NG MADE, IGNORE THIS.
    #THIS WILL WORK BUT IF YOUR FORMATE IS STILL IN .JPG FLATTEND
    w = w.reshape(X.shape[0], 1)
    

    
    A=sigmoid(np.dot(w.T,X)+b); #----THIS WORKS 
    
    #A=sigmoid(np.dot(X,w.T)+b);----THIS Dosnt work why ? 
 
    for i in range(m):
            if(A[0][i]>0.5):
                Y_prediction[0][i]=1;
            else :
                Y_prediction[0][i]=0;
    
    return Y_prediction



 NOTE:
        IF YOUR ACCURACY FOR YOUR TRAINING MODEL IS WAY WAY HIGHER THAN 
        THAT OF YOUR TEST AND CROSS-VALIDATION THEN YOU WILL MOST LIKELY FAIL
        GLOBALIZATION 
        IF THAT IS THE CASE AND SINCE THIS IS A IMAGE DON’T PLAY
        WITH THE FEATURE VECTOR SINCE YOU ALREADY HAVE WHAT IS NEEDED
        FROM FEATURES (RGB * H * W)
        INSTEAD TRY TO PLAY WITH THE LEARNING RATE 
        OR DECREASE THE NUMBER OF ITERATIONS 
        OR TRY TO FIND MUCH, MUCH MORE DIVERSE DATA/IMAGES
        TO REDCUE BIAS AND CALM OVERFITTING
        NOT SURE IF REGULARIZATION IMPLEMENTATION WOULD WORK BUT 
        TRY THAT AS WELL
         
        ALSO BEWARE OF EITHER HIGH VARIANCE OR HIGH BIAS.


def model(X_train, Y_train, X_test, Y_test,X_cross_validation,Y_cross_validation, num_iterations=2000, learning_rate=0.5, print_cost=False):
    #AGAIN USE THE FUNCTION PROVIDED BY DR. ANDREW NG
    w=np.zeros([X_train.shape[0],1]);
    b=0;
    params, grads, costs=optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost);
    w=params["w"];
    b=params["b"];
    Y_prediction_test=predict(w,b,X_test);
    Y_prediction_train=predict(w,b,X_train);
    Y_prediction_cross_validation=predict(w,b,X_cross_validation);

    # Print train/test/CROSS VALIDATION Accuracy
    if print_cost:
        #ACCURACY FOR THE TRAINGING ITS SELF
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100));
        #ACCURACY FOR THE TEST 
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100));
        #ACCURACY FOR THE CROSS VALIDATION 
        print(“cross-validation accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_cross_validation - Y_cross_validation)) * 100));
        
        """
       
        """
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


    logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.001, print_cost=True)

#THIS IS IMPERATIVE FOR AUDITING
costs = np.squeeze(logistic_regression_model['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
plt.show()      
