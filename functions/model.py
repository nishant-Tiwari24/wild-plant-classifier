import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    """
    Wild edible plants classifier containing fully-connected (dense) layers. 
    
    Parameters:
        in_features (int) - number of input nodes (image pixel size)
        out_features (int) - number of output nodes (classes)
        hidden_layers (list) - list of integers to determine the size of the hidden layers
        drop_prob (float) - dropout rate probability
    """
    def __init__(self, in_features, out_features, hidden_layers, drop_prob=0.3):
        super().__init__()
        # First hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(in_features, hidden_layers[0])])
        
        # Add more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # Add output layer
        self.out = nn.Linear(hidden_layers[-1], out_features)
        
        # Set dropout layer to prevent overfitting
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        """
        Used to forward propagate the features through the network. Takes in a tensor of features.
        """
        # Pass each layer through ReLU activation and dropout in between
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        # Output layer with log softmax applied
        x = F.log_softmax(self.out(x), dim=1)
        return x