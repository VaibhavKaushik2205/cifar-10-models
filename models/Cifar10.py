# Import tensorflow libraries
from tf_modules import *

#Building The Model
# Start construction of the Keras Sequential model.
def cifar_model():
    
    # image_shape and no of classes
    input_shape = (32,32,3)
    num_classes = 10
    
    model = Sequential()

    #1st Layer
    model.add(Conv2D(32, kernel_size = 3,padding = 'Same',
                    activation = 'relu',
                    input_shape = input_shape))
    model.add(BatchNormalization())

    #2nd Layer
    model.add(Conv2D(32, kernel_size = 3, padding = 'Same',      
                    activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2,2),
                        strides = 2))
    model.add(Dropout(0.2))

    #3rd Layer
    model.add(Conv2D(64, kernel_size = 3, padding = 'Same',
                    activation = 'relu'))
    model.add(BatchNormalization())

    #4th Layer
    model.add(Conv2D(64, kernel_size = 3, padding = 'Same',
                    activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2,2),
                        strides = 2))
    model.add(Dropout(0.3))
    
    #5th Layer
    model.add(Conv2D(128, kernel_size = 3, padding = 'Same',
                    activation = 'relu'))
    model.add(BatchNormalization())

    #6th Layer
    model.add(Conv2D(128, kernel_size = 3, padding = 'Same',
                    activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2,2),
                        strides = 2))
    model.add(Dropout(0.4))

    #Flatten
    model.add(Flatten())
    
    #Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model