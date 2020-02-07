# Import tensorflow libraries
from tf_modules import *

def conv_module(net, filters, shape, stage, block):
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
    conv_name_base = 'res' + stage + block 
    bn_name_base = 'bn' + stage + block 
    
    # Conv component
    net = Conv2D(filters=filters, kernel_size=(shape,shape),
                 name = conv_name_base ,
                 strides=(1,1), padding='same')(net)
    net = BatchNormalization(axis = 3, name = bn_name_base)(net)
    net = Activation('relu')(net)
    
    return net

def inception_module(net, 
                     filter1x1,
                     filter3x3_reduce,
                     filter3x3,
                     filter5x5_reduce,
                     filter5x5v1,
                     filter5x5v2,
                     filterpool,
                     stage):
        
    # 1x1 branch
    net_1x1 = conv_module(net, filter1x1, shape=1, stage=stage, block='1x1')
    
    # 3x3 branch
    net_3x3 = conv_module(net, filter3x3_reduce, shape=1, stage=stage, block='3x3_downsample')
    net_3x3 = conv_module(net_3x3, filter3x3, shape=3, stage=stage, block='3x3')
    
    # 5x5 branch (use 2 3x3 conv)
    net_3x3v1 = conv_module(net, filter5x5_reduce, shape=1, stage=stage, block='5x5_downsample')
    net_3x3v1 = conv_module(net_3x3v1, filter5x5v1, shape=3, stage=stage, block='3x3v1')
    net_3x3v2 = conv_module(net_3x3v1, filter5x5v2, shape=3, stage=stage, block='3x3v2')
    
    # MaxPool branch
    net_pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(net)
    net_pool = conv_module(net_pool, filterpool, shape=1, stage=stage, block='pool')
    
    # Concatinate all branches together
    net = concatenate([net_3x3, net_3x3v2, net_1x1, net_pool],
                     axis=3, name='inception'+stage )
    
    return net
    
    
def inception():
    """
    Implementation of the ImageNet having the following architecture:
    
    Returns:
    model -- a Model() instance in Keras
    """
    
    # image_shape and no. of classes
    input_shape = (32,32,3)
    num_classes = 10
    
    # Define a tensor with input shape
    inputs = Input(input_shape)
    
    # Stage 1 --> 3x3 Convolution with 16 filters and stride 1
    net = Conv2D(32, (3,3), strides=(1,1), padding='same',
                name='conv1')(inputs)
    net = BatchNormalization(axis=3, name='bn_conv1')(net)
    net = Activation('relu')(net)
    
    # No of inception_modules 
    # Assume 6 modules (loss after each 2 modules)
    
    net = inception_module(net, 
                           filter1x1=32,
                           filter3x3_reduce=24,
                           filter3x3=32,
                           filter5x5_reduce=8,
                           filter5x5v1=16,
                           filter5x5v2=16,
                           filterpool=16,
                           stage='1a')
    
    # Loss after first inception module
    loss1 = AveragePooling2D((3,3),strides=(2,2),name='loss1')(net)
    loss1 = Conv2D(16,(1,1), padding='same',
                        activation='relu',
                        name='loss1-conv')(loss1)
    loss1 = Flatten()(loss1)
    loss1 = Dense(32,activation='relu',name='loss1-fc')(loss1)
    loss1 = Dropout(0.5)(loss1)
    loss1 = Dense(num_classes, name='loss1-classifier',)(loss1)
    loss1 = Activation('softmax')(loss1)
    
    # Continue the NN
    net = inception_module(net, 
                           filter1x1=20,
                           filter3x3_reduce=16,
                           filter3x3=32,
                           filter5x5_reduce=12,
                           filter5x5v1=16,
                           filter5x5v2=16,
                           filterpool=16,
                           stage='2a')
    
    # Loss after 2nd inception module
    loss2 = AveragePooling2D((3,3),strides=(2,2),name='loss2')(net)
    loss2 = Conv2D(16,(1,1), padding='same',
                        activation='relu',
                        name='loss2-conv')(loss2)
    loss2 = Flatten()(loss2)
    loss2 = Dense(32,activation='relu',name='loss2-fc')(loss2)
    loss2 = Dropout(0.5)(loss2)
    loss2 = Dense(num_classes, name='loss2-classifier')(loss2)
    loss2 = Activation('softmax')(loss2)
    
    # Continue the NN
    net = inception_module(net, 
                           filter1x1=32,
                           filter3x3_reduce=20,
                           filter3x3=32,
                           filter5x5_reduce=16,
                           filter5x5v1=32,
                           filter5x5v2=32,
                           filterpool=32,
                           stage='3a')
    
    # Loss of final module
    loss3 = AveragePooling2D((3, 3), strides=(2,2))(net)
    loss3 = Flatten()(loss3)
    loss3 = Dense(128, name='loss3_dense')(loss3)
    loss3 = Activation('relu')(loss3)
    loss3 = Dropout(0.4)(loss3)
    loss3 = Dense(num_classes, name='loss3-classifier')(loss3)
    loss3 = Activation("softmax")(loss3)
    
    model = Model(inputs=inputs, outputs=[loss1,loss2,loss3], name='Inception')
    
    return model
    