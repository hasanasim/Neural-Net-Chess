import numpy as np


def Q_values(x, W1, W2, bias_W1, bias_W2):
    
    """
    FILL THE CODE
    Compute the Q values as ouput of the neural network.
    W1 and bias_W1 refer to the first layer
    W2 and bias_W2 refer to the second layer
    Use rectified linear units
    The output vectors of this function are Q and out1
    Q is the ouptut of the neural network: the Q values
    out1 contains the activation of the nodes of the first layer 
    there are othere possibilities, these are our suggestions
    YOUR CODE STARTS HERE
    """
    
    #reshape x to be usable in calculations
    x = x.reshape(50,1)
    # Activation calculations using relu 
    # input layer to hidden layer 
    act1 = np.dot(W1,x) + bias_W1
    out1 = act1 * (act1>0)

    # hidden layer to output layer 
    act2 = np.dot(W2,out1) + bias_W2
    Q = act2 * (act2>0)

    # YOUR CODE ENDS HERE
    return Q, out1
