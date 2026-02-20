import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        self.sequential = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu, 
            self.fc3
        )
        
    def forward(self, x):
        return self.sequential(x)