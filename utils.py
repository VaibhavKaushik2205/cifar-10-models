'''
Some helper functions:
1. To plot images with labels and predictions
2. Data-Preparation --> Normalize the data (Mean pixel value can also be subtracted)
3. Random-Crop images by first zero padding them to 36x36 then cropping back to 32x32
'''

import numpy as np
import matplotlib.pyplot as plt


# Helper function to Plot Images
def plot_images(images, cls_true, class_names, cls_pred=None, smooth=True ):

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
            
        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
    
# Preparing Data (Normalizing and mean subtraction)
def data_prep(train, validation, test, subtract_pixel_mean):
    # Normalize Data
    train = train.astype('float32')/255
    validation = validation.astype('float32')/255
    test = test.astype('float32')/255
    
    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        train_mean = np.mean(train, axis=0)
        train -= train_mean
        validation -= train_mean
        test -= train_mean
        
    return train, validation, test

# Randomly crop batches of training data
def augment_batches(data, batch_size):
    '''
    Return a total of `num` samples
    '''
    
    images=batch_size
    while(images + batch_size <= len(data)):
        prev_no = images
        images += batch_size
        data[prev_no:images] = crop_images(data[prev_no:images])
    
    return data


# Padding images to 36x36 then randomly crop 32x32 size images
def crop_images(train):
    
    # Zero pad images to 36x36 size
    train = tf.image.resize_with_crop_or_pad(train,
                                             target_height=36,
                                             target_width=36).numpy()
    train = tf.image.random_crop(train, size=[train.shape[0], 32, 32, 3])
    
    return train.numpy()
