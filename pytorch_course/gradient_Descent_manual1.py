import torch
import numpy as np

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)
w = 0.0

def forward(x):
    return w * x

def loss(y, y_predictions):
    return ( ((y_predictions-y)**2 ).mean()) /2

def gradient(x, y, y_predictions):          # dloss/dw
    return np.dot(2*x, y_predictions-y)

print(f'prediction before training: f(5) = {forward(5):.3f}')

learning_rate = 0.01
n_iter = 10

for epoch in range(n_iter):
    y_pred = forward(X)
    L = loss(Y, y_pred)
    dloss_dw = gradient(X,Y, y_pred)
    w -= dloss_dw* learning_rate

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {L:.8f}')

print(f'prediction after training: f(5) = {forward(5):.3f}')




