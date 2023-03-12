def CNN_SeqModel():
    """
    Implements the forward propagation for the binary classification 
model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> 
DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all 
the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the 
entire training process) 
    """
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            
            ## Conv2D with 32 7x7 filters and stride of 1
            
            ## BatchNormalization for axis 3
            
            ## ReLU
            
            ## Max Pooling 2D with default parameters
            
            ## Flatten layer
            
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            
            # YOUR CODE STARTS HERE
        tf.keras.layers.ZeroPadding2D(padding=3,input_shape=(64,64,3)) ,
        tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(7,7),
    strides=(1, 1)
),
        tf.keras.layers.BatchNormalization(
    axis=3,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None),
        tf.keras.layers.ReLU(
    max_value=None, negative_slope=0.0, threshold=0.0
),
           tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(
    data_format=None
),
tf.keras.layers.Dense(
    units=1,
    activation='sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)
            # YOUR CODE ENDS HERE
        ])
    
    return model
