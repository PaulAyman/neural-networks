import torch
import torch.nn as nn
import numpy as np

X_train = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y_train = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)
#y_test   useless
n_sample, n_feature = X_train.shape
input_size = n_feature      #
output_size = n_feature
print(n_sample, n_feature)
learning_rate = 0.01
n_iter = 100

# model = nn.Linear(input_size, output_size)
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):      #constructor
        super(LinearRegression, self).__init__()    #super(): inherit parentClass constructor (nn.Module) ,self: parentClass obj
        self.lin = nn.Linear(input_dim, output_dim) # *core* create obj from nn.Linear 
                                                    # (u can do it 3latoul as above)
    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)  #


print(f'prediction before training: f(5) = {model(X_test).item():.3f}')

for epoch in range(n_iter):
    y_pred = model(X_train)   #== model.forward(X_train) callable obj  that has __cal__ dunder (shorthand in PyTorch)
    l = loss(Y_train, y_pred)   #== loss.forward(Y_train, y_pred) callable obj (shorthand in PyTorch)
 
    l.backward()    # no return!!
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w, b] = model.parameters()     #
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, cost = {l:.8f}')

print(f'prediction after training: f(5) = {model(X_test).item():.3f}')




