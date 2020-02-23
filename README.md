# BigBiGAN-TensorFlow2.0
## Introduction 
An unofficical low-resolution (32 x 32) implementation of BigBiGAN  
Paper: https://arxiv.org/abs/1907.02544
### Architecture
![bigbigan](https://github.com/LEGO999/BIgBiGAN/blob/master/fig/bigbigan.png)
## Dependancy
```
Python 3.6
Tensorflow 2.0.0
Matplotlib 3.1.1
Numpy 1.17
```
## Implementation
### Generator and discriminator
![biggan](https://github.com/LEGO999/BIgBiGAN/blob/master/fig/Screenshot%20from%202020-02-23%2017-02-01.png)
Discriminator F and Generator G come from BigGAN.  
There are 3 residual blocks in G and 4 residual blocks in F.  
Discriminator H und J are 6-layer MLP(units=50) with skip-connection.  
### Encoder
As suggested in the paper, take higher resolution input (64 x 64) and RevNet(13-layers) as backend.  
After RevNet, a 4-layer MLP(Unit=256) is taken.  
The Reversible Residual Network: Backpropagation Without Storing Activations(https://arxiv.org/abs/1707.04585)  
Used implementation: https://github.com/google/revisiting-self-supervised  
## Usage
Set up the flags in ```main.py```. In terminal, enter ```python3 main.py``` to execute the training.
### Supported dataset:
MNIST, Fashion-MNIST and CIFAR10.  
### Conditional und unconditional GAN:
Conditional GAN will use the labels in the corresponding dataset to generate class-specific images.
### Channels und batch size:
Change them to fit in your GPU according to your VRAM(>=6 GB recommended).  
## Known Issue
Conditional GAN is for the time being slightly unstable.
## Examples
MNIST
![mnist](https://github.com/LEGO999/BIgBiGAN/blob/master/fig/mnist1.png)
Fashion-MNIST

