import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


def plot_history(history, title=''):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Model Accuracy {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Model Loss {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def build_model(layers=(128, 64), activation='relu', dropout=0.0, l2=0.0):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for units in layers:
        if l2 > 0:
            model.add(Dense(units, activation=activation, kernel_regularizer=regularizers.l2(l2)))
        else:
            model.add(Dense(units, activation=activation))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))
    return model


results = []

def train_and_record(name, model, optimizer, epochs=10, callbacks=None):
    print(f"\nTraining: {name}")
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2,
                        callbacks=callbacks, verbose=0)
    train_time = time.time() - start
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    row = (name, len(history.history['loss']), history.history['accuracy'][-1],
           history.history['val_accuracy'][-1], test_acc, train_time)
    results.append(row)
    print(f"Epochs: {row[1]}  Train acc: {row[2]:.4f}  Val acc: {row[3]:.4f}  "
          f"Test acc: {row[4]:.4f}  Time: {row[5]:.1f} s")
    return history


# Loading and preprocessing the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Building the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.summary()

# Compiling
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Training and validation curves
plot_history(history, '(Original)')

# Predictions
predictions = model.predict(X_test)

plt.figure(figsize=(5, 5))
plt.imshow(X_test[0], cmap='gray')
plt.title(f"True Label: {np.argmax(y_test[0])}, Predicted: {np.argmax(predictions[0])}")
plt.axis('off')
plt.show()

num_images = 9
plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"True: {np.argmax(y_test[i])}, Predicted: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# Task 1: different architectures and activation functions
print("\n===== Task 1: Different Architectures =====")
architectures = [
    ('Original (128, 64) relu', (128, 64), 'relu'),
    ('Wider (512, 256) relu', (512, 256), 'relu'),
    ('Deeper (256, 128, 64, 32) relu', (256, 128, 64, 32), 'relu'),
    ('Smaller (32,) relu', (32,), 'relu'),
    ('(128, 64) tanh', (128, 64), 'tanh'),
    ('(128, 64) sigmoid', (128, 64), 'sigmoid'),
]

plt.figure(figsize=(8, 6))
for name, layers, activation in architectures:
    h = train_and_record(name, build_model(layers, activation), Adam())
    plt.plot(h.history['val_accuracy'], label=name)
plt.title('Validation Accuracy for Different Architectures')
plt.xlabel('Epochs')
plt.ylabel('Val Accuracy')
plt.legend()
plt.grid()
plt.show()


# Task 2: different optimizers
print("\n===== Task 2: Different Optimizers =====")
optimizers = [('SGD', SGD()), ('RMSprop', RMSprop()), ('Adam', Adam())]

plt.figure(figsize=(12, 5))
for name, opt in optimizers:
    h = train_and_record(f'Optimizer: {name}', build_model(), opt)
    plt.subplot(1, 2, 1)
    plt.plot(h.history['val_accuracy'], label=name)
    plt.subplot(1, 2, 2)
    plt.plot(h.history['loss'], label=name)

plt.subplot(1, 2, 1)
plt.title('Validation Accuracy for Different Optimizers')
plt.xlabel('Epochs')
plt.ylabel('Val Accuracy')
plt.legend()
plt.grid()
plt.subplot(1, 2, 2)
plt.title('Training Loss for Different Optimizers')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# Task 3: overfitting, underfitting, early stopping and regularization
print("\n===== Task 3: Overfitting and Underfitting =====")

# Effect of number of epochs
h = train_and_record('2 epochs', build_model(), Adam(), epochs=2)
plot_history(h, '(2 epochs)')

h = train_and_record('30 epochs', build_model(), Adam(), epochs=30)
plot_history(h, '(30 epochs)')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
h = train_and_record('Early stopping', build_model(), Adam(), epochs=30, callbacks=[early_stop])
plot_history(h, '(Early Stopping)')

# L2 regularization
h = train_and_record('L2 regularization (0.001)', build_model(l2=0.001), Adam())
plot_history(h, '(L2 Regularization)')

# Dropout
h = train_and_record('Dropout (0.3)', build_model(dropout=0.3), Adam())
plot_history(h, '(Dropout)')

# Improved model: dropout + early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
h = train_and_record('Improved: Dropout + Early stopping', build_model(dropout=0.3), Adam(),
                     epochs=30, callbacks=[early_stop])
plot_history(h, '(Improved Model)')


# Summary of all experiments
print("\n===== Summary =====")
print(f"{'Model':38} {'Epochs':>6} {'Train acc':>10} {'Val acc':>8} {'Test acc':>9} {'Time (s)':>9}")
for name, ep, tr, va, te, t in results:
    print(f"{name:38} {ep:6} {tr:10.4f} {va:8.4f} {te:9.4f} {t:9.1f}")
