import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

#-----------------------------------
# MLP is a feedforward network with a 
# configurable number of resnet layers
#-----------------------------------

@dataclass
class MLPConfig:
    n_layers: int = 4
    d_layer: int = 256
    d_input: int = 128
    n_actions: int = 195
    dropout: float = 0.0
    device: str = 'cpu'

class Layer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_layer, config.d_layer),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x):
        return x + self.net(x)

class MLP(nn.Module):

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.device = config.device
        self.input_layer = nn.Linear(config.d_input, config.d_layer)
        self.mid_layers = nn.Sequential(
            *[Layer(config) for _ in range(config.n_layers)]
        )
        self.policy_head = nn.Linear(config.d_layer, config.n_actions)
        self.value_head = nn.Linear(config.d_layer, 1)

        self._init_weights()

        self.to(self.device)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value
    
    def _init_weights(self):
        # Xavier initialization for input and mid layers
        nn.init.xavier_uniform_(self.input_layer.weight)
        for layer in self.mid_layers:
            nn.init.xavier_uniform_(layer.net[0].weight)
        
        # Initialize value heads with smaller weights
        nn.init.xavier_uniform_(self.policy_head.weight, gain=0.01)
        nn.init.xavier_uniform_(self.value_head.weight, gain=0.002)
        
        # Initialize biases to zero
        nn.init.zeros_(self.input_layer.bias)
        for layer in self.mid_layers:
            nn.init.zeros_(layer.net[0].bias)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.bias)
