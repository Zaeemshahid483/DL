import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.metrics import accuracy_score


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        print("Weights:", self.w_)

        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                update = self.eta * (target - y_pred)
                self.w_[1:] = self.w_[1:] + update * xi
                print("Updated Weights:", self.w_[1:])
                self.w_[0] = self.w_[0] + update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self


# Task 3: labels 1 and 0
class PerceptronBinary(Perceptron):

    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, 0)


# Loading data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = shuffle(df)
print(df.head())

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

print(X[0:5])
print(y[0:5])

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25)

train_labels = np.where(train_labels == 'Iris-setosa', 1, -1)
test_labels = np.where(test_labels == 'Iris-setosa', 1, -1)

print('Train data:', train_data[0:2])
print('Train labels:', train_labels[0:2])
print('Test data:', test_data[0:2])
print('Test labels:', test_labels[0:2])

# Training
perceptron = SkPerceptron(eta0=0.1, max_iter=10)
perceptron.fit(train_data, train_labels)

# Testing
test_preds = perceptron.predict(test_data)
print(test_preds)

accuracy = accuracy_score(test_preds, test_labels)
print('Accuracy:', round(accuracy, 2) * 100, '%')


# Task 1: prediction from manual input
print("\nTask 1: Enter flower measurements")
sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))

sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = perceptron.predict(sample)[0]

if prediction == 1:
    print("Prediction: Iris-setosa")
else:
    print("Prediction: Not Iris-setosa")


# Task 3: labels 1 and 0
print("\nTask 3: Labels 1 and 0")
train_labels_01 = np.where(train_labels == 1, 1, 0)
test_labels_01 = np.where(test_labels == 1, 1, 0)

print('Train labels:', train_labels_01[0:10])
print('Test labels:', test_labels_01[0:10])

perceptron_01 = SkPerceptron(eta0=0.1, max_iter=10)
perceptron_01.fit(train_data, train_labels_01)

preds_01 = perceptron_01.predict(test_data)
print(preds_01)
print('Accuracy:', round(accuracy_score(preds_01, test_labels_01), 2) * 100, '%')

my_perceptron = PerceptronBinary(eta=0.01, n_iter=10)
my_perceptron.fit(train_data, train_labels_01)

my_preds = my_perceptron.predict(test_data)
print(my_preds)
print('Accuracy:', round(accuracy_score(my_preds, test_labels_01), 2) * 100, '%')
