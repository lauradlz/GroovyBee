import numpy as np, pandas as pd, glob
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime as dt

from lstm_model import LSTM_Net
from data_loader import load_data
import params

# Define hyperparameters
Epoch_size = 1
sequence_size = 100
step_size = 50
D_in, hidden_dim, D_out = 128, 512, 128 # Neural Network architecture
chunksize = sequence_size * 128

# Build computation graph
model = LSTM_Net(D_in, hidden_dim, D_out)

# Define loss function
loss_fn = nn.modules.loss.MSELoss()

# Define optimizer object
optimizer = optim.Adam(params=model.parameters(), lr = 1e-3)

# Call function to load dataset
# num_sequences is also called batch size
# num_pitches will always be 128
# Dimensions of input_seq will be num_sequences x num_pitches x sequence_length
# Dimensions of target will be num_sequences x num_pitches x 1
midifiles = glob.glob(params.data_loader + params.artist + '/*' + params.instrument + '_*.gz')
file_index = 0
for midifile in midifiles:    
    chunks = pd.read_csv(midifile, compression='gzip', iterator=True, chunksize=chunksize)
    print("reading: " + midifile)

    num_chunks = 0

    for df in chunks:
        input_seq = []
        target = []
        num_chunks += 1

        df = df.T
        # Compute input vector size
        input_range = (df.shape[1] - sequence_size) // step_size

        for i in range(1, input_range, step_size):
            input_seq.append(df.values[1:, i:i+sequence_size])
            target.append(df.values[1:,i+sequence_size:i+sequence_size+1])

        print("Chunk Completed")

        target = np.array(target,dtype=int)

        # Reshape the tensor since PyTorch LSTM expects 3D Tensors
        # LSTM expects input sequence containing 128 elements. 
        # Reshape input to sequence_length x num_sequences x num_pitches
        # Reshape output to 1 x num_sequences x num_pitches
        # dtype converts input python data to compatible PyTorch Tensors 
        dtype = torch.FloatTensor
        input_seq = autograd.Variable(dtype(np.array(input_seq)), requires_grad = True).permute(2,0,1) 
        target = autograd.Variable(dtype(np.array(target)), requires_grad = False).permute(2,0,1) 

        # Define file name for output vectorized midi
        file_name = params.lstm_output + str(file_index) + str(params.instrument) + '_' + params.artist

        # Train the model - 50 iterations will be done over the same dataset
        for t in range(Epoch_size):
            
            # From PyTorch docs - Gradient needs to be reset to 0 after every iteration 
            # to avoid accumulated gradients
            optimizer.zero_grad()

            # Run 1 forward pass on data set
            target_pred = model(input_seq)

            # Compute loss function
            # Returns predictions for every time instance. We only need the last prediction
            # that was computed when all sequences had been passed through LSTM
            loss = loss_fn(target_pred[-1], target)

            # Compute gradients and adjust parameters accordingly
            loss.backward()
            optimizer.step()

            ####### For brevity - comment this out later ########
            print(t, loss.data[0])
            if loss.data[0] < 10:
                break

    # Randomly pick a sequence from input data
    chunk_index = np.random.randint(0,num_chunks)

    chunks = pd.read_csv(midifile, compression='gzip', iterator=True, chunksize=chunksize)

    test_inp = []
    for c, df in enumerate(chunks):
        if c == chunk_index:
            df = df.T
            # Compute input vector size
            input_range = (df.shape[1] - sequence_size) // step_size

            for i in range(1, input_range, step_size):
                test_inp.append(df.values[1:, i:i+sequence_size])
            break

    test_inp = autograd.Variable(dtype(np.array(test_inp)), volatile = True).permute(2,0,1) 

    # Generate new samples
    pred = model(test_inp, future = test_inp.size(0))
    # for i in range(chunksize//num_chunks):
    for i in range(10):
        pred = model(pred, future = test_inp.size(0))

        # Write to CSV
        with open(file_name + '.csv',"a") as f:
            res = pred.type(torch.IntTensor)[-1,:,:]

            df = pd.DataFrame(np.array(res.data))
            df = df.clip(lower=0)
            df.to_csv(f, header=False, index=False)
    file_index += 1
    # # Save the trained model for music generation
    # output_dir = params.lstm_model + params.instrument + '_' + params.artist + '_' + dt.datetime.now().strftime("%y-%m-%d_%H-%M") + '.mod'

    # torch.save(model, output_dir)