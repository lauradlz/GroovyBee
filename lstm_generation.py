import numpy as np
import pandas as pd

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import glob

import params
from lstm_model import LSTM_Net

dtype = torch.FloatTensor

# Load trained model file for music generation
output_dir = glob.glob(params.lstm_model + '*.mod')
f = output_dir[0]

# Apply Model parameters
D_in, hidden_dim, D_out, num_layers = 128, 512, 128, 2
model = LSTM_Net(D_in, hidden_dim, num_layers, D_out, 1)
model = torch.load(f)

# Generate Seed Vector
seed = autograd.Variable(dtype(np.random.randint(127, size=(128)).tolist()), requires_grad = False, volatile=True).view(1,1,-1)

# Generation Algorithm
# Parameter: Track length in ticks
track_length = 200

# Generate first prediction
pred = model(seed).type(torch.IntTensor)

# Define grad param and store first vector of pitches
grid = []
grid.append(pred[:,-1,:].squeeze(0).data.tolist())
last_seq = grid[0]
file_name = params.lstm_output + str(params.instrument) + '_' + params.artist

for j in range(3000):
    for i in range(0,track_length):
        last_seq = model(autograd.Variable(dtype(last_seq), requires_grad = False, volatile=True).view(1,1,-1))
        last_seq = last_seq.type(torch.IntTensor)[:,-1,:].squeeze(0).data.tolist()
        with open(file_name + '.csv',"a") as f:
            df = pd.DataFrame(np.array(last_seq)).T
            df.to_csv(f,header=False,index=False)
    print(j)