import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import os

from config import (
    TRAIN_DATA_PATH,
    MODEL_WEIGHTS_PATH,
    LEARNING_RATE,
    NUM_ITERATIONS,
    DEV_SET_SIZE,
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_CLASSES,
)

def load_data():
    data = pd.read_csv(TRAIN_DATA_PATH)
    data = np.array(data)
    np.random.shuffle(data)

    m, n = data.shape 

    data_dev = data[:DEV_SET_SIZE].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n] / 255.0

    data_train = data[DEV_SET_SIZE:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.0

    return X_train, Y_train, X_dev, Y_dev


def init_params():

    W1 = np.random.rand(HIDDEN_SIZE, INPUT_SIZE) - 0.5
    b1 = np.random.rand(HIDDEN_SIZE, 1) - 0.5

    W2 = np.random.rand(NUM_CLASSES, HIDDEN_SIZE) - 0.5
    b2 = np.random.rand(NUM_CLASSES, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]  
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y                      
    dW2 = (1 / m) * dZ2.dot(A1.T)              
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)       
    dW1 = (1 / m) * dZ1.dot(X.T)              
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2,
                                       dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            preds = get_predictions(A2)
            acc = get_accuracy(preds, Y)
            print(f"iter {i} | train acc: {acc:.4f}")
    return W1, b1, W2, b2


def test_prediction(index, X, Y, W1, b1, W2, b2):
    current_image = X[:, index, None]
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, current_image)
    prediction = get_predictions(A2)[0]
    label = Y[index]
    print("Prediction:", prediction)
    print("Label:", label)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)

def main():

    X_train, Y_train, X_dev, Y_dev = load_data()

    W1, b1, W2, b2 = gradient_descent(
        X_train,
        Y_train,
        LEARNING_RATE,
        NUM_ITERATIONS,
    )

    dev_preds = make_predictions(X_dev, W1, b1, W2, b2)
    dev_acc = get_accuracy(dev_preds, Y_dev)
    print("Final Dev Accuracy:", dev_acc)

    os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
    with open(MODEL_WEIGHTS_PATH, "wb") as f:
        pickle.dump((W1, b1, W2, b2), f)
    print(f"saved weights to {MODEL_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()