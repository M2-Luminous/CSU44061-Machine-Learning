import pandas as pd

# Load the merged data CSV file
merged_data = pd.read_csv('C:/Users/M2-Winterfell/Documents/Code/machine learning/FinalAssignment/database/merged_data_1.csv')

# Ensure 'DATE' is in datetime format
merged_data['DATE'] = pd.to_datetime(merged_data['DATE'])

# Define the splitting date
split_date = pd.Timestamp('2020-03-01')

# Split the DataFrame into two parts
data_before_pandemic = merged_data[merged_data['DATE'] < split_date]
data_during_pandemic = merged_data[merged_data['DATE'] >= split_date]

# Save the two parts to new CSV files
data_before_pandemic.to_csv('C:/Users/M2-Winterfell/Documents/Code/machine learning/FinalAssignment/database/data_before_pandemic.csv', index=False)
data_during_pandemic.to_csv('C:/Users/M2-Winterfell/Documents/Code/machine learning/FinalAssignment/database/data_during_pandemic.csv', index=False)
