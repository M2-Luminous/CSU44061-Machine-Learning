import re
from mido import MidiFile, MidiTrack, Message, MetaMessage

# Define a pitch-to-MIDI mapping (one octave, starting at C=60)
NOTE_TO_MIDI = {
    'C': 60, 'c': 61, 'D': 62, 'd': 63, 'E': 64, 'F': 65, 'f': 66,
    'G': 67, 'g': 68, 'A': 69, 'a': 70, 'B': 71, 'R': None  # R for rest (no pitch)
}

def parse_token(token_str):
    """Parse a single token string into pitch and duration."""
    token_str = token_str.strip()
    if token_str.startswith('[') and token_str.endswith(']'):
        content = token_str[1:-1]  # remove brackets
    else:
        return None, None

    content = re.sub(r'\s+', ' ', content).strip()
    parts = content.split(',')
    if len(parts) != 2:
        return None, None

    pitch = parts[0].strip()
    duration = parts[1].strip().replace(' ', '')

    return pitch, duration

def pitch_to_midi(pitch):
    """Convert a pitch to its corresponding MIDI note number."""
    return NOTE_TO_MIDI.get(pitch, None)

def process_file(file_path):
    """Process the text file to generate a MIDI file."""
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into tokens
    tokens = content.strip().split(']')
    tokens = [t.strip() + ']' for t in tokens if t.strip() != '']

    # Filter out invalid tokens
    tokens = [t for t in tokens if t.startswith('[') and t.endswith(']')]

    # Parse the tokens into a sequence
    parsed_sequence = []
    for tok in tokens:
        p, d = parse_token(tok)
        if p is not None and d.isdigit():
            parsed_sequence.append((p, int(d)))

    # Create a MIDI file and track
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    # Set the tempo to 120 BPM
    track.append(MetaMessage('set_tempo', tempo=500000, time=0))

    velocity = 64
    current_time = 0

    for pitch, duration in parsed_sequence:
        midi_note = pitch_to_midi(pitch)
        if pitch == 'R' or midi_note is None:
            # Rest: advance the time by the duration
            current_time += duration
        else:
            # Note: add note_on and note_off messages
            track.append(Message('note_on', note=midi_note, velocity=velocity, time=current_time))
            track.append(Message('note_off', note=midi_note, velocity=velocity, time=duration))
            current_time = 0  # Reset current_time after each event

    # Save the MIDI file
    output_file = 'output.mid'
    mid.save(output_file)
    print(f"MIDI file '{output_file}' created successfully.")

# Process the uploaded file
file_path = 'C:/Users/M2-Winterfell/Downloads/ML/generated_melody.txt'
process_file(file_path)
