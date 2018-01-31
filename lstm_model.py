import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define custom LSTM Class
# Required because PyTorch method of chaining NN layers has issues with LSTMs
class LSTM_Net(nn.Module):
    def __init__(self, D_in, hidden_dim, D_out):
        super(LSTM_Net,self).__init__()
        # Assign Hyperparameter values to class variables
        self.hidden_dim = hidden_dim

        # Build LSTM NN with 2 Layers using dropout regularizer
        self.lstm_1 = nn.LSTMCell(D_in, hidden_dim)
        self.dropout = nn.modules.Dropout(p=0.2)
        # 2nd Layer for LSTM
        self.lstm_2 = nn.LSTMCell(hidden_dim, hidden_dim)
        # Add Linear Densely connected NN to transform 512 LSTM nodes to a vector of 128 outputs (pitches)  
        self.linear = nn.Linear(hidden_dim, D_out)
        # Activation Layer
        self.activation = nn.Linear(D_in, D_out)

    # Define computation graph for Forward pass
    def forward(self, x, future = 0):
        outputs = []
        h_1 = autograd.Variable(torch.zeros(x.size(0), self.hidden_dim))
        c_1 = autograd.Variable(torch.zeros(x.size(0), self.hidden_dim))
        h_2 = autograd.Variable(torch.zeros(x.size(0), self.hidden_dim))
        c_2 = autograd.Variable(torch.zeros(x.size(0), self.hidden_dim))

        for i,input_x in enumerate(x.chunk(x.size(1), dim=1)):
            # LSTM returns cell state and hidden state at time t
            h_1, c_1 = self.lstm_1(input_x.squeeze(1), (h_1, c_1))
            d = self.dropout(h_1)
            h_2, c_2 = self.lstm_2(d, (h_2, c_2))
            # We use total hidden state as input to Linear Layer
            y_pred = self.linear(h_2)
            # Activation Function
            output = self.activation(y_pred)
            # Return predictions
            outputs +=[output]
        
        for i in range(future):
            # LSTM returns cell state and hidden state at time t
            h_1, c_1 = self.lstm_1(y_pred, (h_1, c_1))
            d = self.dropout(h_1)
            h_2, c_2 = self.lstm_2(d, (h_2, c_2))
            # We use total hidden state as input to Linear Layer
            y_pred = self.linear(h_2)
            # Activation Function
            output = self.activation(y_pred)
            # Return predictions
            outputs += [output]
        
        outputs = torch.stack(outputs,1).squeeze(2)
        return outputs