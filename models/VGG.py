# Import tensorflow libraries
from tf_modules import *

def vgg_model(input_shape, architecture, num_classes):
    # Initialize the model along with the input shape to be
    # "channels last" ordering
    # Here, we are taking the size of all filters as 3x3 and adding BatchNormalisation after each layer
    # Architecture -- list telling architecture of VGG model (eg. for VGG11->[64,M,128,M,256,256,M,512,512,M])
    
    inputs = Input(input_shape)
    net = inputs
    
    for i in (architecture):
        if i=='M':
            net = MaxPooling2D(pool_size = 2, strides = 2,
                          padding="same")(net)
        else:
            net = Conv2D(filters=i, kernel_size=(3,3),
                         dtype=tf.float32, strides=1,
                         padding='same')(net)
            net = Activation("relu")(net)
            net = BatchNormalization()(net)
        
    # Flatten
    net = Flatten()(net)
    
    # First Dense Layer with ReLU activation
    net = Dense(512)(net)
    net = Activation("relu")(net)
    net = Dropout(0.4)(net)

    # Final Dense Layer with softmax activation
    net = Dense(num_classes)(net)
    net = Activation("softmax")(net)
    
    model = Model(inputs=inputs, outputs=net, name='VGG')
    # return the constructed network architecture
    return model


def vgg(model):
    
    # image_shape and no. of classes
    input_shape = (32,32,3)
    num_classes = 10
    
    # VGG Models
    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    
    return vgg_model(input_shape, cfg[model], num_classes)
