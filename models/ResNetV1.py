# Import tensorflow libraries
from tf_modules import *

'''
# Define an Identity Block
The identity block is the standard block used in ResNets, and corresponds to the case where the input activationhas the same dimension as the output activation
'''

def identity_block(net, shape, filters, stage, block):
    '''
    Arguments: 
    1. net -- input tensor of shape(m, h_prev, w_prev, c_prev)
    2. shape -- integer, specifying the shape of the middle CONV's
    window for the main path
    3. filters -- list of integers, defining the no of filters of 
    CONV layers in the main path
    4. stage -- integer, used to name the layers depending on their 
    position in the network
    5. block -- string/character used to name the layers depending
    on their position in the network
    
    Returns:
    net -- output of the identity block, tensor of shape
    (h_curr, w_curr, c_curr)
    '''
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'
    
    # Retrieve filters
    # we are using two conv layers hence
    F1, F2 = filters
    
    # Save input value to be used as shortcut to add to main path
    net_shortcut = net
    
    ## Main Path
    # First component
    net = Conv2D(filters=F1, kernel_size=(shape,shape),
                 name = conv_name_base + '2a',
                 strides=(1,1), padding='same')(net)
    net = BatchNormalization(axis = 3, name = bn_name_base + '2a')(net)
    net = Activation('relu')(net)
    
    ## Second component
    net = Conv2D(filters=F2, kernel_size=(shape,shape),
                 name = conv_name_base + '2b',
                 strides=(1,1), padding='same')(net)
    net = BatchNormalization(axis = 3, name = bn_name_base + '2b')(net)
    
    # Final step: Add shortcut value to main path, 
    # and pass it through a RELU activation
    
    net = Add()([net, net_shortcut])
    net = Activation('relu')(net)
    
    return net

'''
# Convolution Block
This is used when the output the dimension is different than the input dimension in a layer
This block is used when the input and output dimensions don't match up. The difference with the identity block is that there is a CONV2D layer in the shortcut path which is used to resize the net to a different dimension, so that the dimensions add up for the final addition when the shortcut is added to the main path
'''

def convolutional_block(net, shape, filters, stage, block, s=2):
    '''
    Arguments: 
    1. net -- input tensor of shape(m, h_prev, w_prev, c_prev)
    2. shape -- integer, specifying the shape of the middle CONV's
    window for the main path
    3. filters -- list of integers, defining the no of filters of 
    CONV layers in the main path
    4. stage -- integer, used to name the layers depending on their 
    position in the network
    5. block -- string/character used to name the layers depending
    on their position in the network
    6. s -- Integer specifying the stride to be used
    
    Returns:
    net -- output of the identity block, tensor of shape
    (h_curr, w_curr, c_curr)
    '''
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'
    
    # Retrieve filters
    # we are using two conv layers hence
    F1, F2 = filters
    
    # Save input value to be used as shortcut to add to main path
    net_shortcut = net
    
    ## Main Path
    # First component
    net = Conv2D(filters=F1, kernel_size=(shape,shape),
                 name = conv_name_base + '2a',
                 strides=(s,s), padding='same')(net)
    net = BatchNormalization(axis = 3, name = bn_name_base + '2a')(net)
    net = Activation('relu')(net)
    
    ## Second component
    net = Conv2D(filters=F2, kernel_size=(shape,shape),
                 name = conv_name_base + '2b',
                 strides=(1,1), padding='same')(net)
    net = BatchNormalization(axis = 3, name = bn_name_base + '2b')(net)
    
    ## Shortcut Path
    net_shortcut = Conv2D(filters=F2, kernel_size=(1,1), strides=(s,s),
                         padding='valid',
                         name = conv_name_base + '1')(net_shortcut)
    net_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(net_shortcut)
    
    # Final step: Add shortcut value to main path, 
    # and pass it through a RELU activation
    
    net = Add()([net, net_shortcut])
    net = Activation('relu')(net)
    
    return net


## ResNet 
def resnet(depth):
    """
    Implementation of the ResNet20 having the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> IDBLOCK*2 -> 
    IDBLOCK*2 -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    depth -- No of convolutions layers in the network (including start and end layers)
    
    Returns:
    model -- a Model() instance in Keras
    """
    
    # image_shape and no. of classes
    input_shape = (32,32,3)
    num_classes = 10
    
    if (depth-2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        
    # Define a tensor with input shape
    inputs = Input(input_shape)
    
    # Stage 1 --> 3x3 Convolution with 16 filters and stride 1
    net = Conv2D(16, (3,3), strides=(1,1), padding='same',
                name='conv1')(inputs)
    net = BatchNormalization(axis=3, name='bn_conv1')(net)
    net = Activation('relu')(net)
    
    num_blocks_per_layer = int((depth-2)/6)
    num_filters = 16
    
    for stage in range(3):
        block = 0
        
        while block < num_blocks_per_layer:
            
            if stage>0 and block==0: # First block of later layers which need downsampling
                net = convolutional_block(net, shape=3, filters=[num_filters,num_filters], stage=stage, block=block, s=2)
                block += 1
                
            net = identity_block(net, shape=3, filters=[num_filters,num_filters], stage=stage, block=block)
            block += 1
            
        num_filters *= 2
            
    # Global Average Pooling
    net = AveragePooling2D((2,2), name="avg_pool")(net)
    
    # Flatten Layer
    net = Flatten()(net)
    
    # Output Layer
    net = Dense(num_classes, activation='softmax', name='fc')(net)
    
    # Create Model
    model = Model(inputs=inputs, outputs=net, name='ResNet{}'.format(depth))
    
    return model
    
    