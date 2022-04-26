# Imports python modules
from torch import nn
import torch.nn.functional as F

class Classifier(nn.Module):
    """ Create custom classifier for use with pretrained model
    """
    def __init__(self, input_units, hidden_units, output_units, drop_rate):
        super().__init__()
        
        self.hidden = nn.ModuleList([nn.Linear(input_units, hidden_units[0])])
        layer_units = zip(hidden_units[:-1], hidden_units[1:])
        self.hidden.extend([nn.Linear(h1, h2) for h1, h2 in layer_units])
        
        self.output = nn.Linear(hidden_units[-1], output_units)
        
        self.dropout = nn.Dropout(p=drop_rate)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        for h in self.hidden:
            x = F.relu(h(x))
            x = self.dropout(x)
            
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        
        return x