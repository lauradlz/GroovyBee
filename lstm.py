import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import load_data

dtype = torch.FloatTensor

Epoch_size = 50
Sequence_size = 12
Step_size = 6

# Define Neural Network architecture
D_in, L_1_out, L_2_in, L_2_out, D_out = 128, 512, 512, 512, 128

model = nn.Sequential(
    nn.LSTM(D_in, L_1_out)
    # nn.Dropout(p=0.5),
    # nn.LSTM(L_2_in, L_2_out),
    # nn.Linear(L_2_out, D_out)
)

# Define loss function
loss_fn = nn.CrossEntropyLoss()

# Define optimizer object
optimizer = optim.Adam(params=model.parameters(), lr = 1e-4)

# Call function to load dataset
input_seq, target = load_data(Sequence_size, Step_size)

input_seq = autograd.Variable(dtype(input_seq))
target = autograd.Variable(dtype(target))

# Reshape the tensor since PyTorch LSTM expects 3D Tensors
input_seq = input_seq.permute(2,0,1)
target = target.permute(2,0,1)

# Train the model
for t in range(Epoch_size):

    optimizer.zero_grad()

    # Run 1 forward pass on data set
    target_pred = model(input_seq)

    # Compute loss function
    loss = loss_fn(target_pred, target)
    
    ####### For brevity - comment this out later ########
    print(t, loss.data[0])

    # Compute gradients and adjust parameters accordingly
    loss.backward()
    optimizer.step()
