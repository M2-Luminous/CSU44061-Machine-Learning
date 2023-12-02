import os
import pandas as pd
import glob
# https://www.visualcrossing.com/weather/weather-data-services

# Define the path where your CSV files are located on your local machine
path = '../FinalAssignment/database'

# Use glob to match the naming pattern of your CSV files
file_pattern1 = os.path.join(path, 'dublinbikes_*.csv')
file_pattern2 = os.path.join(path, 'dublinbike-historical-data-*.csv')

# Get a list of all CSV files that match the pattern
csv_files = glob.glob(file_pattern1) + glob.glob(file_pattern2)

# Initialize an empty list to store DataFrames
dataframes_list = []

# Loop through the list of filenames and read each file into a DataFrame
for filename in csv_files:
    df = pd.read_csv(filename)
    dataframes_list.append(df)

# Concatenate all the DataFrames into one
concatenated_df = pd.concat(dataframes_list, ignore_index=True)

# Preprocessing steps (if needed)
# For example, if you want to convert the 'TIME' column to datetime format you could do:
# concatenated_df['TIME'] = pd.to_datetime(concatenated_df['TIME'])

# Save the concatenated DataFrame into a new CSV file
output_filename = os.path.join(path, 'aggregated_data.csv')
concatenated_df.to_csv(output_filename, index=False)

# The path to the new aggregated file will be:
print(f'Aggregated file saved at: {output_filename}')
