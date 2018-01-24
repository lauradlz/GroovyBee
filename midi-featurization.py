import numpy as np
import pandas as pd
import os, sys, glob
import midi

# def trackToMidi(track, midifile):
#     t = []
#     t.append(track)
#     pattern = midi.Pattern(t)
#     midi.write_midifile("C:/Users/Pc Laura/Desktop/Deep_Learning/Mini_Project/GroovyBee/" + midifile +"-track1.mid",  pattern)
#     return

def trackToFeature(track):
    totalTicks = 0
    musicalEvents = []
    instrument = -1
    for event in track:
        # If the event is not a Meta Event, must be a Musical Event
        if isinstance(event, midi.MetaEvent):
            continue
        else:
            totalTicks += event.tick
            musicalEvents.append(event)
    
    # Implies the track contained only Meta Information
    if totalTicks == 0:
        return None, None
    
    grid = np.zeros((128, totalTicks))
    current_vector = np.zeros(128)
    pos = 0

    for event in musicalEvents:
        if isinstance(event, midi.NoteEvent):
            if event.tick != 0:
                for i in range(0,event.tick):
                    grid[:,pos] = current_vector
                    pos += 1
            if isinstance(event, midi.NoteOffEvent):
                current_vector[event.pitch] = 0
            if isinstance(event, midi.NoteOnEvent):
                current_vector[event.pitch] = event.velocity             
        else:
            pos += event.tick
            # Get value of played instrument
            if isinstance(event, midi.ProgramChangeEvent):
                instrument = event.value
    return instrument, grid

def remove_zeros(feature_vec):
    # Sum along columns. Silence periods will contains '0' Velocity for all 128 Pitches
    # Consecutive vectors of sum 0 will ensure removal of first vector containing zeros
    feature_vec_sum = feature_vec.sum(axis=0)
    zero_indices = []

    # Gets indices of columns where sum along all pitches is 0
    for i in range(1, len(feature_vec_sum)):
        if feature_vec_sum[i] == feature_vec_sum [i-1] == 0:
            zero_indices.append(i-1)
    
    # Returns a new copy of feature vector without the consecutive zero vectors
    return np.delete(feature_vec, zero_indices, axis=1)

def main():
    artist = sys.argv[1] 
    midifiles = glob.glob('./data/*.mid')
    for midifile in midifiles:   
        audio = midi.read_midifile(midifile)
        instrument_indices = np.zeros(len(audio), dtype=int)

        for index,track in enumerate(audio):
            # Featurize a single track from the audio file
            instrument, feature_vec = trackToFeature(track)
            if feature_vec is not None:
                feature_vec_sanitized = remove_zeros(feature_vec)
                df = pd.DataFrame(feature_vec_sanitized)
                file_name = './output/' + str(instrument) + '_' + artist + '_'

                # Check if file exists, and set index appropriately
                if os.path.isfile(file_name + str(instrument_indices[index]) + '.gz'):
                    instrument_indices[index] += 1
                # Write file to output folder
                df.to_csv(file_name + str(instrument_indices[index]) + '.gz', compression='gzip')
            
if __name__ == "__main__":
    main()