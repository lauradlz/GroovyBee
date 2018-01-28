import midi

def trackToMidi(track, midifileName):
    eot = midi.EndOfTrackEvent(tick = 1) #essential for the track to be turned into a readable midi file
    track.append(eot)
    pattern = midi.Pattern(track)
    midi.write_midifile("C:/Users/Pc Laura/Desktop/Deep_Learning/Mini_Project/GroovyBee/" + midifileName +".mid",  pattern)
    return

# Go through the fist column to fin the first not to be played (assuming columns for ticks and roaw for 128 differents notes)
def decoder(grid):
    track = []
    previousVector = grid[0]
    # Search for first note in the grid
    for noteIndex in previousVector:
        if previousVector[noteIndex] != 0:
            on = midi.NoteOnEvent(0, previousVector[noteIndex], noteIndex) #Add NoteOn event at tick = 0, velocity = previousVector[noteIndex], pitch = noteIndex
            track.append(on)
    # Turn off notes and add following        
    tickOffset = 0
    for vector in grid:
        if previousVector == vector:
            tickOffset += 1
        else:
            for noteIndex in previousVector:
                if previousVector[noteIndex] == vector[noteIndex]:
                    continue
                if previousVector[noteIndex] != 0 and vector[noteIndex] = 0:
                    off = midi.NoteOffEvent(tickOffset, noteIndex)
                    track.append(off)
                else:
                    on = midi.NoteOnEvent(0, vector[noteIndex], noteIndex)
                    track.append(on)
                tickOffset = 0
            tickOffset += 1
        previousVector = vector

    return track


def main():
    grid = outputLSTM
    #decode output grid from LTSM into one full track of notes:
    decoder(grid)
    #translate track into midi file
    trackToMidi(track, nameofmidifile)

if __name__ == "__main__":
    main()

