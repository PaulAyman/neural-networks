import torch
import numpy as np

def forward(w, x):
    return w * x

def loss(y, y_predictions):
    return y_predictions-y

def cost(y, y_predictions):
    L = loss(y, y_predictions)
    return ( (L**2).mean() )    # mean = sum/m (yufadal no /2)

learning_rate = 0.01
n_iter = 1000
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32 ,requires_grad=True)

print(f'prediction before training: f(5) = {forward(w, 5):.3f}')

for epoch in range(n_iter):
    y_pred = forward(w, X)
    c = cost(Y, y_pred)
    # dcost_dw = gradient(X,Y, y_pred)

    c.backward()    # no return!!
    with torch.no_grad():
        w -= w.grad * learning_rate
    w.grad.zero_()
    
    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, cost = {c:.8f}')

print(f'prediction after training: f(5) = {forward(w, 5):.3f}')




