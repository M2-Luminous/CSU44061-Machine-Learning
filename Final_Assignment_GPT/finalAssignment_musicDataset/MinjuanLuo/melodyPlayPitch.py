import re
from mido import MidiFile, MidiTrack, Message
from mido import MetaMessage


# Example output line (as provided):
line = "[ A , 0 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ D , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ g , 5 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ D , 9 ]   [ B , 9 ]   [ C , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ R , 3 1 2 ]   [ B , 3 2 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ a , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ c , 9 ]   [ c , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ c , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ A , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ D , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ D , 9 ]   [ c , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ B , 9 ]   [ E , 9 ]"

# Define a pitch-to-MIDI mapping (one octave, starting at C=60)
NOTE_TO_MIDI = {
    'C':60, 'c':61, 'D':62, 'd':63, 'E':64, 'F':65, 'f':66,
    'G':67, 'g':68, 'A':69, 'a':70, 'B':71, 'R':None  # R for rest (no pitch)
}

def parse_token(token_str):
    # Remove brackets
    token_str = token_str.strip()
    if token_str.startswith('[') and token_str.endswith(']'):
        content = token_str[1:-1]  # remove leading '[' and trailing ']'
    else:
        return None, None

    # Normalize spaces
    content = re.sub(r'\s+', ' ', content).strip()
    parts = content.split(',')
    if len(parts) != 2:
        return None, None

    pitch = parts[0].strip()
    duration = parts[1].strip()

    # Remove any internal spaces from pitch/duration
    pitch = pitch.replace(' ', '')
    duration = duration.replace(' ', '')

    return pitch, duration

def pitch_to_midi(pitch):
    # Return MIDI number or None if it's a rest
    return NOTE_TO_MIDI.get(pitch, None)

# Parse the entire line of tokens
tokens = line.strip().split(']')
tokens = [t.strip() + ']' for t in tokens if t.strip() != '']  # re-add ']' to each token
# This step reconstitutes tokens because splitting by ']' lost that character
# Alternatively, you could split differently or ensure tokens are well-defined.

# Clean any trailing incorrect tokens
tokens = [t for t in tokens if t.startswith('[') and t.endswith(']')]

parsed_sequence = []
for tok in tokens:
    p, d = parse_token(tok)
    if p is not None and d is not None:
        # Convert duration to int
        try:
            dur_val = int(d)
        except ValueError:
            # If parsing failed, skip token or set a default
            dur_val = 480  # default duration
        parsed_sequence.append((p, dur_val))

# Create a MIDI track from the parsed sequence
mid = MidiFile(ticks_per_beat=480)  # 480 ticks per quarter note
track = MidiTrack()
mid.tracks.append(track)

# Set a tempo (120 BPM)
# 500000 microseconds per quarter note
track.append(MetaMessage('set_tempo', tempo=500000, time=0))

velocity = 64
current_time = 0

# We'll interpret duration as the time until the next event.
for (pitch, duration) in parsed_sequence:
    midi_note = pitch_to_midi(pitch)
    if pitch == 'R' or midi_note is None:
        # This is a rest. Just advance time by 'duration' ticks without playing a note.
        current_time += duration
    else:
        # This is a note
        # Note on at current_time (since last event)
        # We'll encode the note as note_on with time=0 then note_off with time=duration
        # Add a note_on at current_time offset from previous event
        track.append(Message('note_on', note=midi_note, velocity=velocity, time=current_time))
        # Note_off after 'duration' ticks
        track.append(Message('note_off', note=midi_note, velocity=velocity, time=duration))
        # Reset current_time to 0 because we've accounted for the passage of time
        current_time = 0

# Save the MIDI file
mid.save('output.mid')

# Playing the MIDI file:
# Mido doesn't provide direct audio playback. You can open 'output.mid' in a MIDI player or DAW.
# If you want direct playback from Python:
# 1. Install pygame: pip install pygame
# 2. Use pygame.mixer.music to load and play the file:
# 
# import pygame
# pygame.init()
# pygame.mixer.music.load("output.mid")
# pygame.mixer.music.play()
# while pygame.mixer.music.get_busy():
#     pass
#
# However, the above is platform-dependent and requires a MIDI synthesizer.
#
# The simplest approach: just open "output.mid" in your preferred MIDI player.

print("MIDI file 'output.mid' created. Please open it in a MIDI player to hear the melody.")
