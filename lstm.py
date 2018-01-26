import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import load_data

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

# Define hyperparameters
Epoch_size = 50
Sequence_size = 9
Step_size = 6
D_in, hidden_dim, D_out, num_layers = 128, 512, 128, 2 # Neural Network architecture

# Call function to load dataset
# num_sequences is also called batch size
# num_pitches will always be 128
# Dimensions of input_seq will be num_sequences x num_pitches x sequence_length
# Dimensions of target will be num_sequences x num_pitches x 1
input_seq, target = load_data(Sequence_size, Step_size) 

# Reshape the tensor since PyTorch LSTM expects 3D Tensors
# LSTM expects input sequence containing 128 elements. 
# Reshape input to sequence_length x num_sequences x num_pitches
# Reshape output to 1 x num_sequences x num_pitches
# dtype converts input python data to compatible PyTorch Tensors 
dtype = torch.FloatTensor
input_seq = autograd.Variable(dtype(input_seq)).permute(2,0,1) 
target = autograd.Variable(dtype(target)).permute(2,0,1) 

# Build computation graph
model = LSTM_Net(D_in, hidden_dim, num_layers, D_out, input_seq.size(1))

# Define loss function
loss_fn = nn.modules.loss.MSELoss()

# Define optimizer object
optimizer = optim.Adam(params=model.parameters(), lr = 1e-3)

# Train the model - 50 iterations will be done over the same dataset
for t in range(Epoch_size):
    
    # From PyTorch docs - Gradient needs to be reset to 0 after every iteration 
    # to avoid accumulated gradients
    optimizer.zero_grad()

    # Reset memory of LSTM hidden layer
    model.hidden = model.init_hidden()
    
    # Run 1 forward pass on data set
    target_pred = model(input_seq)

    # Compute loss function
    # Returns predictions for every time instance. We only need the last prediction
    # that was computed when all sequences had been passed through LSTM
    loss = loss_fn(target_pred[-1], target)

    # Compute gradients and adjust parameters accordingly
    loss.backward(retain_graph=True)
    optimizer.step()

    ####### For brevity - comment this out later ########
    print(t, loss.data[0])