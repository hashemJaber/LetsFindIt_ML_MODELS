#THIS IS THE TITANIC PROBLEM, GIVEN A MALE HE SHOULD NOT GET IN THE LIFEBOAT, A FEMALE ON THE OTHER HAND MUST GET IN, THIS VERY SIMPLE LOGISTIC REGRESSION MODEL WILL SHOW US GENERALLY HOW NEURAL NETWORKS WORK EXCEPT WITHOUT BACKWARD PROPAGATION TO OTHER NUERAL, ACTUALLY, THIS IS A SINGLE NEURON IN A SINGLE LAYER, ITS HARD TO CALL IT A NEURAL NETWORK BUT REGARDLESS THIS ALL WORK THE SAME WAY JUST WITH DIFFERENT ACTIVATION FUNCTIONS AND OTHER CURRENTLY NON IMPORTANT DETAILS WICH IS NOT NEEDED FOR A PROBLEM WITH SUCH LOW COMPLEXITY, ALSO THIS PROBLEM WILL BE SOLVED WITHOUT VICTORIZATION 


import numpy as np;
number_of_nuerons=4;
number_of_inputs=2;
def layer_initilizer(number_of_nuerons=1,number_of_inputs=1):
    
    return np.random.randn(number_of_nuerons,number_of_inputs)*0.01;
w=np.array([[1,1]])
print(w);

x=np.array([[1,0],[0,1]]);
y=np.array([[0,1]]);

def sigmoid(z):
    return 1/(1+np.exp(-z));

predictions=sigmoid(np.dot(x,w.T));

print("predictions : ",predictions);

def correct(x,y,w,rate=0.001,iterations=200):
    
    for i in range(iterations):
        new_w1=0;
        new_w2=0;
        predictions=sigmoid(np.dot(x,w.T));
        new_w1=w[0][0]-(rate*(x[0][0]*(y[0][0]-predictions[0][0]))  );
        new_w2=w[0][1]-(rate*(x[0][1]*(y[0][0]-predictions[0][0]))  );
        w[0][0]=new_w1;
        w[0][1]=new_w2;
        
        
        new_w1=0;
        new_w2=0;
      
        new_w1=w[0][0]-(rate*(x[1][0]*(y[0][1]-predictions[1][0]))  );
        new_w2=w[0][1]-(rate*(x[1][1]*(y[0][1]-predictions[1][0]))  );
        w[0][0]=new_w1;
        w[0][1]=new_w2;
        
        
        
  
    return w
print("weights before correction", w);    
w=correct(x,y,w,rate=12,iterations=2000);  
print("weights after correction", w);   

predictions=sigmoid(np.dot(x,w.T));

print("predictions after correction: ",predictions);

print("so given a male (0,1) we should get 0 ",sigmoid(np.dot(np.array([[0,1]]),w.T)) );
#RESULTS:<img width="1338" alt="Screen Shot 2023-01-15 at 12 50 50 AM" src="https://user-images.githubusercontent.com/91439032/212531822-918ac201-9d6b-479d-996b-60bfff08cfe2.png">
