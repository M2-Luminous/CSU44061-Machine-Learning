# -*- coding: utf-8 -*-
"""
Augment melodies by applying controlled pitch shifts.
Preserves structured text representation with [Pitch, Duration] tokens.
"""

# Define note mappings
NOTES = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']

# Function to shift notes while preserving durations
def translate_notes(sequence, shift):
    tokens = sequence.split(' ')
    translated = []
    for token in tokens:
        if token.startswith('[') and token.endswith(']'):
            pitch, duration = token.strip('[]').split(',')
            if pitch in NOTES:  # Apply shift to notes
                new_pitch = NOTES[(NOTES.index(pitch) + shift) % len(NOTES)]
                translated.append(f"[{new_pitch},{duration}]")
            else:  # Preserve rests or invalid tokens
                translated.append(token)
    return ' '.join(translated)

# Load input melodies
with open('inputMelodies.txt', 'r') as file:
    input_melodies = file.readlines()

# Apply upward shifts (configurable if needed)
shifts = [1, 2, 3, 4, 5]  # Keep shifts simple for harmonic integrity
augmented_melodies = []

for shift in shifts:
    for melody in input_melodies:
        melody = melody.strip()
        if melody:
            augmented_melodies.append(translate_notes(melody, shift))

# Save augmented melodies
with open('inputMelodiesAugmented.txt', 'w') as file:
    for melody in augmented_melodies:
        file.write(melody + "\n")

print("The augmented melodies have been saved to inputMelodiesAugmented.txt.")
