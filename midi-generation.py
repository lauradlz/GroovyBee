import midi
import glob, pandas as pd
import params
import numpy as np

# Go through the fist column to fin the first not to be played (assuming columns for ticks and roaw for 128 differents notes)
def decoder(track, grid):
    previousVector = grid[0]
    # Search for first note in the grid
    for noteIndex, note in enumerate(previousVector):
        if note != 0:
            on = midi.NoteOnEvent(tick = 0, velocity = int(max(127,min(note,0))), pitch = noteIndex) #Add NoteOn event at tick = 0, velocity = previousVector[noteIndex], pitch = noteIndex
            track.append(on)
    # Turn off notes and add following        
    tickOffset = 0
    for vector in grid[1:]:
        if (previousVector == vector).all():
            tickOffset += 1
        else:
            for noteIndex, note in enumerate(previousVector):
                if note == vector[noteIndex]:
                    continue
                if note != 0 and vector[noteIndex] == 0:
                    off = midi.NoteOffEvent(tick = tickOffset, pitch = noteIndex)
                    track.append(off)
                else:
                    on = midi.NoteOnEvent(tick = 0, velocity =  int(max(127,min(vector[noteIndex],0))), pitch = noteIndex)
                    track.append(on)
                tickOffset = 0
            tickOffset += 1
        previousVector = vector

    return track

def postprocess(df):
    pitches = np.max(df.values, axis=0)
    df_copy = df
    for index,col in enumerate(df.values.T):
        if pitches[index] == 0:
            continue
        col = col / pitches[index]
        col[col>1.1] = 0
        col[col<0.9] = 0
        col = col * pitches[index]
        df_copy[index] = col
    return df_copy

def main():
    midifiles = glob.glob(params.lstm_output + '/0*.csv')
    
    # Define midi pattern parameters
    pattern = midi.Pattern()
    pattern.resolution = 240

    for midifile in midifiles:
        track = midi.Track()
        track.append(midi.SetTempoEvent(bpm = np.random.randint(50,72)))

        df = pd.read_csv(midifile, header=None)
        df = postprocess(df)
        # Decode output grid from LTSM into one full track of notes:
        track = decoder(track, df.values[:,0:])
        eot = midi.EndOfTrackEvent(tick = 1) #essential for the track to be turned into a readable midi file
        track.append(eot)
        # Translate track into midi file
        pattern.append(track)

    midi.write_midifile(params.midi_generated + params.artist + ".mid",  pattern)
    
if __name__ == "__main__":
    main()

