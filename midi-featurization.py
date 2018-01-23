import midi
import numpy as np

def getFirstTrack(midifile):
    audio = midi.read_midifile(midifile)
    track = audio[1]
    return track

def trackToMidi(track, midifile):
    t = []
    t.append(track)
    pattern = midi.Pattern(t)
    midi.write_midifile("C:/Users/Pc Laura/Desktop/Deep_Learning/Mini_Project/GroovyBee/" + midifile +"-track1.mid",  pattern)
    return

def trackToFeature(track):
    totalTicks = 0
    musicalEvents = []
    for event in track:
        if isinstance(event, midi.NoteEvent):
            totalTicks += event.tick
            musicalEvents.append(event)
    grid = np.zeros((128, totalTicks))
    current_vector = np.zeros(128)
    pos = 0
    # for event in musicalEvents:
    #     if isinstance(event, NoteOnEvent)
    #     if event.tick != 0:
    #         for i in event.tick:

    return 

def main():
    midifilenames = ['tchop35a']
    ext = ".mid"
    for midifile in midifilenames:   
        track = getFirstTrack('C:/Users/Pc Laura/Desktop/Deep_Learning/Mini_Project/GroovyBee/data/' + midifile + ext)
        trackToMidi(track, midifile)
        # print(trackToFeature(track))  
            
if __name__ == "__main__":
    main()