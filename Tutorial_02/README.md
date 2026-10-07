# Tutorial 2 – MLP Classifier

## Objective
Train a Multi-Layer Perceptron (MLP) on the Iris dataset to classify all three flower species. The tutorial covers scaling the data with StandardScaler, training and evaluating the model, reading the classification report, and plotting the learning (loss) curve.

## Files
`Tutorial_2.py` contains the full tutorial code and both tasks.

## Tasks
**Task 1 – Different hidden layers and neurons:** The model was trained with eight configurations, from one layer of 5 neurons up to three layers of 100 neurons, and their accuracy, epochs and loss curves were compared.

| Hidden layers | Accuracy | Epochs | Final loss |
|---|---|---|---|
| (5,) | 1.00 | 1000 | 0.1390 |
| (10,) | 1.00 | 1000 | 0.0947 |
| (50,) | 1.00 | 591 | 0.0735 |
| (10, 10) | 1.00 | 609 | 0.0718 |
| (50, 50) | 1.00 | 353 | 0.0461 |
| (10, 10, 10) | 1.00 | 857 | 0.0318 |
| (50, 50, 50) | 0.98 | 397 | 0.0047 |
| (100, 100, 100) | 1.00 | 239 | 0.0031 |

Since Iris is a small and easy dataset, almost every configuration reached 100% accuracy. The main difference was in the learning curve: more layers and neurons made the loss drop faster and reach a lower value. Small networks such as (5,) and (10,) did not converge within 1000 epochs. Very large networks reach almost zero training loss, which can lead to overfitting, as seen with (50, 50, 50) at 0.98. **(50, 50)** gave the best balance: 100% accuracy in 353 epochs with a smooth curve.

**Task 2 – Different learning rates:** The (10, 10) model was trained with five learning rates and the loss curves were plotted.

| Learning rate | Accuracy | Epochs | Final loss |
|---|---|---|---|
| 0.0001 | 0.73 | 1000 | 0.5507 |
| 0.001 | 1.00 | 609 | 0.0718 |
| 0.01 | 1.00 | 157 | 0.0494 |
| 0.1 | 0.96 | 159 | 0.0011 |
| 0.5 | 1.00 | 73 | 0.0542 |

A very small learning rate (0.0001) learns too slowly and did not converge in 1000 epochs. Increasing the learning rate makes the loss fall faster and reduces the number of epochs. However, large learning rates make training unstable: with 0.5 the loss jumped to about 7 in the first epochs before coming down, and 0.1 dropped to 96% accuracy. **0.01** gave the best balance, reaching 100% accuracy in 157 epochs with a smooth loss curve.

## How to Run
Run `python Tutorial_2.py`, or in Google Colab upload the file and run `%run Tutorial_2.py`.

## Requirements
Python 3, scikit-learn, Matplotlib
