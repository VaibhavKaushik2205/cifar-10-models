# Import tensorflow libraries
from tf_modules import *

def alexnet():
    # Initialize the model along with the input shape to be
    # "channels last" ordering
    # Note: in Alexnet, all paddings were 'valid'
    # Here, we are taking the size of all filters as 3x3 and adding BatchNormalisation after each layer
    
    # image_shape and num_classes (pre-defined)
    input_shape = (32,32,3)
    num_classes = 10
    
    model = Sequential()
    
    # Define the first CONV Layer
    model.add(Conv2D(48, (3,3), # In Alexnet this kernel size is 11
                     input_shape=input_shape,
                     name='conv_layer1',
                     dtype=tf.float32,
                     strides=1, ## Originally: 4
                     padding='same')) ## Orignally:'valid'
    model.add(Activation("relu", name='activation1'))
    model.add(MaxPooling2D(pool_size = 2, strides = 2,
                          padding="same", name='maxpool1'))
    model.add(BatchNormalization(name='BatchNorm1'))
    
    # Define the second CONV Layer
    model.add(Conv2D(96, (3, 3),
                     name='conv_layer2',
                     dtype=tf.float32,
                     padding='same')) 
    model.add(Activation("relu", name='activation2'))
    model.add(MaxPooling2D(pool_size = 2, strides = 2,
                          padding='same', name='maxpool2'))
    model.add(BatchNormalization(name='BatchNorm2'))
    
    # Define the third CONV Layer
    model.add(Conv2D(192, (3, 3), padding='same',
                     name='conv_layer3',
                     dtype=tf.float32))
    model.add(Activation("relu", name='activation3'))
    model.add(BatchNormalization(name='BatchNorm3'))
    
    # Define the fourth CONV Layer
    model.add(Conv2D(192, (3, 3), padding='same',
                     name='conv_layer4',
                     dtype=tf.float32))
    model.add(Activation("relu", name='activation4'))
    model.add(MaxPooling2D(pool_size = 2, strides = 2,
                          padding='same', name='maxpool3'))
    
    model.add(BatchNormalization(name='BatchNorm4'))
    
    ## Define fifth CONV Layer
    model.add(Conv2D(256, (3, 3), padding='same',
                     name='conv_layer5',
                     dtype=tf.float32))
    model.add(Activation("relu", name='activation5'))
    model.add(MaxPooling2D(pool_size = 2, strides = 2,
                          padding='same', name='maxpool4'))
    model.add(BatchNormalization(name='BatchNorm5'))
    
    # First Dense Layer with ReLU activation
    model.add(Flatten())
    
    # Second Dense Layer with ReLU activation
    model.add(Dense(512, name='dense1'))
    model.add(Activation("relu", name='activation6'))
    model.add(Dropout(0.4))

    # Third Dense Layer with ReLU activation
    model.add(Dense(256, name='dense2'))
    model.add(Activation("relu", name='activation7'))
    model.add(Dropout(0.4))

    # Final Dense Layer with softmax activation
    model.add(Dense(num_classes, name='dense3'))
    model.add(Activation("softmax", name='activation8'))
    
    # return the constructed network architecture
    return model
