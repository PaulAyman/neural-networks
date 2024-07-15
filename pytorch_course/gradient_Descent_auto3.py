import torch
import torch.nn as nn
import numpy as np

learning_rate = 0.01
n_iter = 100
X_train = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y_train = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

n_sample, n_feature = X_train.shape
input_size = n_feature
output_size = n_feature
print(n_sample, n_feature)


model = nn.Linear(input_size, output_size)
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)


print(f'prediction before training: f(5) = {model(X_test).item():.3f}')

for epoch in range(n_iter):
    y_pred = model(X_train)
    # c = cost(Y, y_pred)
    l = loss(Y_train, y_pred)
 
    l.backward()    # no return!!
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, cost = {l:.8f}')

print(f'prediction after training: f(5) = {model(X_test).item():.3f}')




