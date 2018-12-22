# -*- coding: utf-8 -*-
"""
Based on Hiroyuki.Konno DDPG agent version
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    input_size = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(input_size)
    return (-lim, lim)

class Actor(nn.Module):
    """ Actor (Policy) Model """
    
    def __init__(self, state_size, action_size, seed, hidden_layers=[64,64]):
        """ Initilize parameters and build model
        Params
        ======
            state_size (int): Dimension of state
            action_size (int): Dimension of action
            seed (int): Random seed
            hidden_layers list(int): hidden layer size
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.input_batchnorm = nn.BatchNorm1d(state_size)

        dims = [state_size] + hidden_layers
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.batchnorm_layers = nn.ModuleList([nn.BatchNorm1d(dim_out) for dim_out in dims[1:]])

        self.out_layer = nn.Linear(dims[-1], action_size)
        self.output_batchnorm = nn.BatchNorm1d(action_size)

        self.reset_parameters()
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions"""

        x = self.input_batchnorm(state)
        for layer, batch_norm in zip(self.layers, self.batchnorm_layers):
            x = layer(x)
            x = batch_norm(x)
            x = F.relu(x)
            
        x = self.out_layer(x)
        x = torch.tanh(x)
        return x
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
            
        self.out_layer.weight.data.uniform_(*hidden_init(self.out_layer))
        
    
class Critic(nn.Module):
    """ Critic (Value) Model """
    
    def __init__(self, state_size, action_size, seed, hidden_layers=[64, 64]):
        """ Initilize parameters and build model
        Params
        ======
            state_size (int): Dimension of state
            action_size (int): Dimension of action
            seed (int): Random seed
            hidden_layers list(int): hidden layer size (have to be equal or 
                                     more than 2 layers)
        """
        super(Critic, self).__init__()
        assert len(hidden_layers) > 1
        self.seed = torch.manual_seed(seed)
        
        self.input_batchnorm = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        
        dims = [hidden_layers[0]+action_size, ] + hidden_layers[1:]
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        
        self.out_layer = nn.Linear(dims[-1], 1)

        self.reset_parameters()
        
    def forward(self, state, action):
        """ Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        xs = self.input_batchnorm(state)
        xs = F.relu(self.fc1(xs))
        x = torch.cat((xs, action), dim=1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            
        x = self.out_layer(x)
        
        return x
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.out_layer.weight.data.uniform_(-3e-3, 3e-3)
            
        