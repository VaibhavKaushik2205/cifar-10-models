# Train CIFAR-10 with TensorFlow

I trained various architectures on the CIFAR-10 dataset using functional model

***

| Model       | Description            |
|-------------|------------|
|  Cifar-10     |        A simple convolutionl model with 7 Layers     |
| AlexNet      |    A Convolutonal Network with 8 layers        |
|   VGG    |     An implementation of VGG11, VGG 13, VGG 16, VGG 19       |
| Inception | InceptionV2 with 2 3x3 convolutions stacked (instead of a 5x5 Conv) | 
| ResNet (Convolutional Layers) | An implementation of ResNet model having convolutional blocks | 
| ResNet (BottleNeck Layers) | An implementation of ResNet model having bottleneck blocks (i.e. 1x1 Conv followed by a 3x3 Conv) |
| DenseNet | An implementation of DenseNet architecture having 4 Dense Blocks and 3 Transition Blocks (DenseNet121, DenseNet169, DenseNet201 and DenseNet161 with growth-rate 12) |

***

### Data-Preprocessing

Data normalization with per-pixel mean subtracted

***

### Learning rate adjustment and optimization

- I trained the models for 300 epochs and manually adjusted the learning-rate
The initial learning rate is set to 0.1, and is divided by 10 at 50% and 75% of the total number of training epochs

- The networoks are trained using stochastic gradient descent (SGD) or Adam(not effective in some architectures) using a weight decay of 0.0001 and a Nesterov momentum of 0.9 