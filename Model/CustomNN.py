import torch
import torch.nn as nn

class CustomNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_p, activation):
        super(CustomNN, self).__init__()
        self.layers = nn.ModuleList()
        
        #Input layer:
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.layers.append(activation)
        self.layers.append(nn.Dropout(p=dropout_p))
        
        #Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(activation)
            self.layers.append(nn.Dropout(p=dropout_p))
            
        #Output layers
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))
        
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x