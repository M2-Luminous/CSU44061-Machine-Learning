# -*- coding: utf-8 -*-
"""
Improved MIDI to text processing script for melodies.
Includes rest handling and supports nested folder traversal.
"""

import os
from mido import MidiFile, MidiTrack, Message

# Define paths
source_dir = 'C:/Users/M2-Winterfell/Downloads/ML/CSU44061-Machine-Learning/Final_Assignment_GPT/finalAssignment_musicDataset/BiMMuDa/bimmuda_dataset/'  # Root dataset folder
simplified_dir = 'musicDatasetSimplified'

# Create output directory if it doesn't exist
if not os.path.exists(simplified_dir):
    os.makedirs(simplified_dir)

# Note mappings
MIDI_NOTE_TO_NAME = {0: 'C', 1: 'c', 2: 'D', 3: 'd', 4: 'E', 5: 'F', 6: 'f', 7: 'G', 8: 'g', 9: 'A', 10: 'a', 11: 'B'}
NOTES = list(MIDI_NOTE_TO_NAME.values())

# Traverse nested folders to locate .mid files (excluding `_full.mid`)
def find_midi_files(root):
    midi_files = []
    for year in os.listdir(root):
        year_path = os.path.join(root, year)
        if os.path.isdir(year_path):
            for rank in os.listdir(year_path):
                rank_path = os.path.join(year_path, rank)
                if os.path.isdir(rank_path):
                    for file in os.listdir(rank_path):
                        if file.endswith('.mid') and not file.endswith('_full.mid'):  # Exclude `_full.mid`
                            midi_files.append(os.path.join(rank_path, file))
    return midi_files

# Convert MIDI to text sequence
def midi_to_text_sequence(midi_path):
    midi = MidiFile(midi_path)
    sequence = []
    last_tick_time = 0  # Track the timing of the last processed note
    
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note = MIDI_NOTE_TO_NAME.get(msg.note % 12, '')  # Map MIDI note to note name
                if note:
                    # Calculate rest duration
                    rest_duration = msg.time - last_tick_time
                    if rest_duration > 0:  # Add rest token if needed
                        sequence.append(f"[R,{rest_duration}]")
                    
                    # Add the note
                    sequence.append(f"[{note},{msg.time}]")
                    last_tick_time = msg.time  # Update last tick time

    return ' '.join(sequence)

# Convert text sequence back to MIDI
def text_sequence_to_midi(sequence, output_path):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    tokens = sequence.split(' ')
    
    for token in tokens:
        if token.startswith('[') and token.endswith(']'):
            pitch, duration = token.strip('[]').split(',')
            duration = int(duration)
            if pitch == 'R':
                track.append(Message('note_off', note=0, velocity=0, time=duration))
            else:
                midi_note = list(MIDI_NOTE_TO_NAME.keys())[list(MIDI_NOTE_TO_NAME.values()).index(pitch)]
                midi_note += 12 * 5  # Adjust to mid-range octave
                track.append(Message('note_on', note=midi_note, velocity=64, time=0))
                track.append(Message('note_off', note=midi_note, velocity=64, time=duration))
    
    midi.save(output_path)

# Process all MIDI files and generate text sequences
midi_files = find_midi_files(source_dir)
text_sequences = []
for midi_path in midi_files:
    sequence = midi_to_text_sequence(midi_path)
    if sequence:
        text_sequences.append(sequence)

# Save text sequences to file
with open("inputMelodies.txt", "w") as file:
    for sequence in text_sequences:
        file.write(sequence + "\n")

# Convert text sequences back to MIDI
for i, sequence in enumerate(text_sequences):
    output_path = os.path.join(simplified_dir, f"output_midi_{i+1}.mid")
    text_sequence_to_midi(sequence, output_path)

print("Simplified text sequences and MIDI files generated.")
