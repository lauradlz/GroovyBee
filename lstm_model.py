import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define custom LSTM Class
# Required because PyTorch method of chaining NN layers has issues with LSTMs
class LSTM_Net(nn.Module):
    def __init__(self, D_in, hidden_dim, num_layers, D_out, num_sequences):
        super(LSTM_Net,self).__init__()
        # Assign Hyperparameter values to class variables
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_sequences = num_sequences

        # Build LSTM NN with 2 Layers using dropout regularizer
        self.lstm = nn.LSTM(D_in, hidden_dim, num_layers, dropout=0.2)
        # Add Linear Densely connected NN to transform 512 LSTM nodes to a vector of 128 outputs (pitches)  
        self.linear = nn.Linear(hidden_dim, D_out)
        # Assign hidden layer shared variables, for gradient computations (backward pass)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.num_layers, self.num_sequences, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers, self.num_sequences, self.hidden_dim)))

    # Define computation graph for Forward pass
    def forward(self,x):
        # LSTM returns cell state and hidden state at time t
        out, self.hidden = self.lstm(x, self.hidden)
        # We use total hidden state as input to Linear Layer
        y_pred = self.linear(out)
        # Return predictions
        return y_pred