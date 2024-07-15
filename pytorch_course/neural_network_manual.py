import torch
import torch.nn as nn

class MyNeuralNetwork(nn.Module):
    # layer dimensions
    def __init__(self, input_size, hidden_size, n_classes):
        super(MyNeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, n_classes)
        # self.sigmoid = nn.Sigmoid()

    # layer training
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        #  y_pred = self.sigmoid(out)   OR
        #  y_pred = torch.sigmoid(out)
        return out
    
model = MyNeuralNetwork(input_size=28*28, hidden_size=5, n_classes=3)
criterion = nn.CrossEntropyLoss()   # (softmax layer)
# OR nn.MSELoss     OR nn.BCELoss 
#     (linear)         (logistic)      
