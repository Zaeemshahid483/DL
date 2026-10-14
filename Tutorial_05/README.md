# Tutorial 5 – Convolutional Neural Network (CNN)

## Objective
Build, train and evaluate a Convolutional Neural Network that classifies the 10 classes of the CIFAR-10 dataset (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The tutorial covers how convolution and pooling layers extract features from images, how to plot training and validation curves, and how to improve accuracy and reduce overfitting.

The tutorial was given in TensorFlow/Keras and has been implemented here in **PyTorch**. The original model is identical to the tutorial's: three convolution layers (32, 64, 64 filters), two max-pooling layers, a 64-neuron dense layer and a 10-class output, with 122,570 parameters in both versions.

## Files
| File | Description |
|---|---|
| `Tutorial_5.py` | Complete code, split into numbered blocks |
| `Tutorial_5_Colab.ipynb` | The same blocks as a Colab notebook |

## Tasks
**Task 1 – Different architectures:** The original model was compared with four modified versions, each trained for 10 epochs: a larger 5×5 filter size, more filters per layer (64, 128, 128), an extra convolution layer (32, 64, 64, 128), and more fully connected layers (256, 128).

**Task 2 – Improving accuracy:** An improved model was built using two convolution layers per block with batch normalization, more filters (32 → 64 → 128), and data augmentation (random crops and horizontal flips) on the training images.

**Task 3 – Removing overfitting:** The original model overfits: training accuracy keeps rising while validation loss starts increasing after a few epochs. The improved model reduces this using dropout after every block, weight decay (L2 regularization), data augmentation and early stopping, which restores the weights from the epoch with the lowest validation loss.

## How to Run
In Google Colab, select **Runtime → Change runtime type → T4 GPU**, then either upload `Tutorial_5_Colab.ipynb` (File → Upload notebook) and run all cells, or paste each block of `Tutorial_5.py` into its own cell.

## Requirements
Python 3, PyTorch, torchvision, NumPy, Matplotlib (all preinstalled in Google Colab)
