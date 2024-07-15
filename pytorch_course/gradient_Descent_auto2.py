import torch
import torch.nn as nn
import numpy as np

def forward(w, x):
    return w * x

# def loss(y, y_predictions):
#     return y_predictions-y

# def cost(y, y_predictions):
#     L = loss(y, y_predictions)
#     return ( (L**2).mean() )    # mean = sum/m (yufadal no /2)

learning_rate = 0.01
n_iter = 100
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32 ,requires_grad=True)

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr= learning_rate)

print(f'prediction before training: f(5) = {forward(w, 5):.3f}')

for epoch in range(n_iter):
    y_pred = forward(w, X)
    # c = cost(Y, y_pred)
    l = loss(Y, y_pred)

    l.backward()    # no return!!
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, cost = {l:.8f}')

print(f'prediction after training: f(5) = {forward(w, 5):.3f}')




