# -*- coding: utf-8 -*-
"""
Extract _full.mid files from nested directory structure.
Copies files to a flat target directory.
"""

import os
import shutil

# Define paths
source_dir = 'C:/Users/M2-Winterfell/Downloads/ML/CSU44061-Machine-Learning/Final_Assignment_GPT/finalAssignment_musicDataset/BiMMuDa/bimmuda_dataset/'  # Original dataset location
target_dir = 'musicDatasetOriginal'

# Create target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Traverse the nested directory structure and copy "_full.mid" files
for year in os.listdir(source_dir):
    year_path = os.path.join(source_dir, year)
    if os.path.isdir(year_path):
        for rank in os.listdir(year_path):  # Rank folders: "1", "2", "3", "4", "5"
            rank_path = os.path.join(year_path, rank)
            if os.path.isdir(rank_path):
                for file in os.listdir(rank_path):
                    if file.endswith('_full.mid'):  # Look for '_full.mid' files
                        full_file_path = os.path.join(rank_path, file)
                        shutil.copy(full_file_path, target_dir)

print(f"All '_full.mid' files copied to {target_dir}.")
