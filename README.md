# Train CIFAR-10 with TensorFlow

I trained various architectures on the CIFAR-10 dataset using functional model

***

| Model       | Description            |
|-------------|------------|
|  Cifar-10     |        A simple convolutionl model with 7 Layers     | 83.42% |  |
| [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)      |    A Convolutonal Network with 8 layers        | 
|   [VGG](https://arxiv.org/pdf/1409.1556.pdf)    |     An implementation of VGG11, VGG 13, VGG 16, VGG 19       | 
| [Inception](https://arxiv.org/pdf/1409.4842.pdf) | InceptionV2 with 2 3x3 convolutions stacked (instead of a 5x5 Conv) | 
| [ResNet (Convolutional Layers)](https://arxiv.org/pdf/1512.03385.pdf) | An implementation of ResNet model having convolutional blocks |
| [ResNet (BottleNeck Layers)](https://arxiv.org/pdf/1512.03385.pdf) | An implementation of ResNet model having bottleneck blocks (i.e. 1x1 Conv followed by a 3x3 Conv) |  
| [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) | An implementation of DenseNet architecture having 4 Dense Blocks and 3 Transition Blocks (DenseNet121, DenseNet169, DenseNet201 and DenseNet161 with growth-rate 12) |

***

### Data-Preprocessing 
Data normalization with per-pixel mean subtraction

***

### Learning rate adjustment and optimization 
I trained the models for 300 epochs and manually adjusted the learning-rate
- lr for epochs [0,150) = 0.1
- lr for epochs [150,225) = 0.01
- lr for epochs {225,300) = 0.01

The networoks are trained using stochastic gradient descent (SGD) or Adam(not effective in some architectures) using a weight decay of 0.0001 and a Nesterov momentum of 0.9 