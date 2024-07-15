import torch
import numpy as np

def forward(w, x):
    return w * x

def loss(y, y_predictions):
    return y_predictions-y

def cost(y, y_predictions):
    L = loss(y, y_predictions)
    return ( (L**2).mean() )/2    # mean = sum/m

def gradient(x, y, y_predictions):          # d(cost)/dw = dcost/dloss * dloss/dy_pred * dy_pred/dw 
    L = loss(y, y_predictions)
    return (np.dot(x, L)).mean()

learning_rate = 0.01
n_iter = 100
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)
w = 0.0

print(f'prediction before training: f(5) = {forward(w, 5):.3f}')

for epoch in range(n_iter):
    y_pred = forward(w, X)
    c = cost(Y, y_pred)
    dcost_dw = gradient(X,Y, y_pred)
    w -= dcost_dw * learning_rate

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, cost = {c:.8f}')

print(f'prediction after training: f(5) = {forward(w, 5):.3f}')




