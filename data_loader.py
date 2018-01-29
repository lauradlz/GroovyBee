import numpy as np
import pandas as pd
import glob
import params 

def load_data(sequence_size, step_size):
    midifiles = glob.glob(params.data_loader + params.artist + '/*' + params.instrument + '_*.gz')
    
    input_seq = []
    target = []
    
    # for midifile in midifiles:
    #     df = pd.read_csv(midifile, compression='gzip')
    #     print("reading: " + midifile)
    #     # Compute input vector size
    #     input_range = (df.shape[1] - sequence_size) // step_size

    #     for i in range(0,input_range,step_size):
    #         input_seq.append(df.values[:, i:i+sequence_size].ravel().reshape(128,sequence_size))
    #         target.append(df.values[:,i+sequence_size:i+sequence_size+1].ravel().reshape(128,1))

    # print("Completed")
    # return input_seq, np.array(target,dtype=int)

    # -------------To avoid memory error
    # -------------Dont run in final version
    midifiles = midifiles[0]
    
    chunks = pd.read_csv(midifiles, compression='gzip', iterator=True, chunksize=1000)
    print("reading: " + midifiles)
    for df in chunks:
        # Compute input vector size
        input_range = (df.shape[1] - sequence_size) // step_size

        for i in range(1,input_range,step_size):
            input_seq.append(df.values[:, i:i+sequence_size].ravel().reshape(128,sequence_size))
            target.append(df.values[:,i+sequence_size:i+sequence_size+1].ravel().reshape(128,1))

        print("Chunk Completed")
    return input_seq, np.array(target,dtype=int)