# Import tensorflow libraries
from tf_modules import *

# Dense Block
def dense_block(net, depth, growth_rate, in_features, block):
    '''
    Arguments:
    depth -- no of dense layers in a dense block (i.e. 16)
    in_features -- no of output feature maps of previous dense block
    block --  current block 
    '''

    # Save current layer to concatenate with produced feature-map
    input_layer = net
    for i in range(depth):
        
        # 1x1 Convolution to reduce size
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = Conv2D(4*growth_rate, kernel_size=(1,1), strides=(1,1),
                     padding='valid')(net)
        net = Dropout(0.2)(net)
        
        # 3x3 Convolutions 
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = Conv2D(growth_rate, kernel_size=(3,3), strides=(1,1),
                     padding='same')(net)
        net = Dropout(0.2)(net)
        
        # Concatinate previous layer and output feature-map
        net = concatenate([input_layer,net], axis=-1)

        input_layer = net
                
        # No of outpzt feature maps of dense block
        out_features = growth_rate*depth + in_features
    
    return net, out_features
        
# Transition Block
def transition_block(net, n, in_features, block):
    '''
    Arguments:
    n -- compression_factor to reduce no of input feature maps
    in_features -- no of output feature maps of previous dense block
    block --  current block 
    '''
    # No of output feature maps of transition block
    out_features = int(n*in_features)
    
    net = BatchNormalization(name='bn_transition_block'+str(block))(net)
    net = Activation('relu')(net)
    net = Conv2D(out_features, kernel_size=(1,1), strides=(1,1),
                 padding='same', name='Conv_transition_block'+str(block))(net)
    net = Dropout(0.2)(net)
    
    # Avg pooling 
    net = AveragePooling2D((2,2))(net)
    
    return net, out_features
    
def densenet_model(growth_rate, compression, depth):
    """
    Implementation of the DenseNet 
    Architecture of a each of Conv layer: BN -> ReLU -> Conv -> Dropout(0.2)
    
    Arguments:
    blocks -- no of dense blocks
    growth_rate -- no of feature maps produced by each layer
    depth -- array containing the no of layers in a dense block

    Returns:
    model -- a Model() instance in Keras
    """
    
    # image_shape and no. of classes
    input_shape = (32,32,3)
    num_classes = 10
    
    #No of layers in transition block
    n=compression
    
    # Define a tensor with input shape
    inputs = Input(input_shape)
    
    # Stage 1 --> 3x3 Convolution with 24 filters and stride 1
    net = Conv2D(24, (3,3), strides=(1,1), padding='same',
                name='conv1')(inputs)
    
    # Input features to first dense_block
    num_features = 24
    num_blocks = 4
    
    # Create a dense block followed by a transition block
    for block in range(num_blocks):
        if (block+1)==num_blocks: # Last block
            # No transition layer after last block
            net, num_features = dense_block(net=net, 
                                            depth=depth[block],
                                            growth_rate=growth_rate,
                                            in_features=num_features,
                                            block=block+1)
            
        else:
            net, num_features = dense_block(net=net, 
                                            depth=depth[block],
                                            growth_rate=growth_rate,
                                            in_features=num_features, 
                                            block=block+1)
            
            net, num_features = transition_block(net=net,
                                                 n=compression,
                                                 in_features=num_features,
                                                 block=block+1)
    
    net = BatchNormalization(name='bn_conv_last')(net)
    net = Activation('relu')(net)
    net = AveragePooling2D((2,2), strides=(2,2))(net)
    
    #Flatten layer
    net = Flatten()(net)
    net = Dense(num_classes, name='Dense-layer')(net)
    net = Activation('softmax')(net)
    
    # Create Model
    model = Model(inputs=inputs, outputs=net, name='DenseNet')
    
    return model

def densenet(model, growth_rate, compression):
    
    architecture = {'densenet121':[6,12,24,16],
                    'densenet169':[6,12,32,32],
                    'densenet201':[6,12,48,32],
                    'densenet161':[6,12,36,24]
                   }
    # return model
    return densenet_model(growth_rate, compression, architecture[model])

