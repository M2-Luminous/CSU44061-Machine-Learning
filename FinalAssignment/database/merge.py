import pandas as pd

# File paths
weather_data_path = "C:/Users/M2-Winterfell/Documents/Code/machine learning/FinalAssignment/database/weather_after_pandemic.csv"
predata_path = "C:/Users/M2-Winterfell/Documents/Code/machine learning/FinalAssignment/database/after_pandemic.csv"

# Load the CSV files into DataFrames
weather_data = pd.read_csv(weather_data_path)
predata = pd.read_csv(predata_path)

# Convert the date columns to datetime format if they are not already
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
predata['DATE'] = pd.to_datetime(predata['DATE'])

# Merge the DataFrames on the date column
# Assuming 'datetime' in weather_data corresponds to 'DATE' in predata
merged_data = pd.merge(predata, weather_data, left_on='DATE', right_on='datetime', how='left')

# Drop the 'datetime' column from the merged data
merged_data.drop('datetime', axis=1, inplace=True)

# Save the merged data to a new CSV file
merged_data.to_csv("C:/Users/M2-Winterfell/Documents/Code/machine learning/FinalAssignment/database/data_after_pandemic.csv", index=False)
