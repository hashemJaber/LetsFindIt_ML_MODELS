def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> 
DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some 
values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the 
entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    # Z1 = None
    
   
    # YOUR CODE STARTS HERE
    Z1=tf.keras.layers.Conv2D(
    filters=8,
    kernel_size=(4,4),
    strides=(1, 1),
    padding='SAME',
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(input_img);
    
    ## RELU
    # A1 = None
    A1=tf.keras.layers.ReLU(
    max_value=None, negative_slope=0.0, threshold=0.0
)(Z1);
    
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    # P1 = None
    P1=tf.keras.layers.MaxPool2D(
    pool_size=(8, 8),
    strides=8,
    padding='SAME'
)(A1);
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    # Z2 = None
    Z2=tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=(2,2),
    strides=(1, 1),
    padding='SAME',
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(P1);
    ## RELU
    # A2 = None
    A2=tf.keras.layers.ReLU(
    max_value=None, negative_slope=0.0, threshold=0.0
)(Z2);
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # P2 = None
    P2=tf.keras.layers.MaxPool2D(
    pool_size=(4, 4),
    strides=4,
    padding='SAME'
)(A2);
    ## FLATTEN
    # F = None
    F=tf.keras.layers.Flatten()(P2);
     ## Dense layer
    
    ## 6 neurons in output layer. Hint: one of the arguments should be 
"activation='softmax'" 
    # outputs = None
    outputs= tf.keras.layers.Dense(
    units=6,
    activation='softmax',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(F);
    
    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
