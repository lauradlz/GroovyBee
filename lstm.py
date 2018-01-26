import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import load_data

class LSTM_Net(nn.Module):
    def __init__(self, D_in, hidden_dim, num_layers, D_out, num_sequences):
        super(LSTM_Net,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_sequences = num_sequences

        self.lstm = nn.LSTM(D_in, hidden_dim, num_layers, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, D_out)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.num_layers, self.num_sequences, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers, self.num_sequences, self.hidden_dim)))

    def forward(self,x):
        out, self.hidden = self.lstm(x, self.hidden)
        y_pred = self.linear(out)
        return y_pred

dtype = torch.FloatTensor

Epoch_size = 50
Sequence_size = 12
Step_size = 6

# Call function to load dataset
input_seq, target = load_data(Sequence_size, Step_size) # 32x128x12 : 32 Sequences, 128 Rows, 12 Sequence length

# Reshape the tensor since PyTorch LSTM expects 3D Tensors
input_seq = autograd.Variable(dtype(input_seq)).permute(2,0,1) # 12x32x128 : 12 Sequence length (12 columns), 32 Mini batch size, 128 input elements (data points)
target = autograd.Variable(dtype(target)).permute(2,0,1) # 1x32x128 : Output label vector has size 1 (1 column)

# Define Neural Network architecture
D_in, hidden_dim, D_out, num_layers = 128, 512, 128, 2

model = LSTM_Net(D_in, hidden_dim, num_layers, D_out, 1)

# Define loss function
loss_fn = nn.modules.loss.MSELoss()

# Define optimizer object
optimizer = optim.Adam(params=model.parameters(), lr = 1e-4)

# Train the model - 50 iterations will be done over the same dataset
for t in range(Epoch_size):
    loss = None
    # Input dimension contains number of sequences 
    for i in range(input_seq.shape[1]):
        optimizer.zero_grad()

        # Reset memory of LSTM
        model.hidden = model.init_hidden()

        # Run 1 forward pass on data set
        target_pred = model(input_seq[:,i,:])

        # Compute loss function
        loss = loss_fn(target_pred[-1], target[:,i,:].squeeze(0))

        # Compute gradients and adjust parameters accordingly
        loss.backward()
        optimizer.step()
    ####### For brevity - comment this out later ########
    print(t, loss.data[0])