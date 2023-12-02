import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'C:/Users/M2-Winterfell/Documents/Code/machine learning/FinalAssignment/database/data_before_pandemic.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Calculate BUR and BAI
df['BUR'] = df['AVAILABLE BIKES'] / df['BIKE STANDS']
df['BAI'] = df['AVAILABLE BIKE STANDS'] / df['BIKE STANDS']

# Calculate CUAS and round to 3 decimal places
df['CUAS'] = (df['BUR'] - df['BAI'])
df['BIKE USAGE'] = df['CUAS'].round(decimals=2)

# Drop the BUR and BAI columns
df.drop(['BUR', 'BAI', 'CUAS'], axis=1, inplace=True)

# Save the updated DataFrame to a new CSV file
output_file = 'C:/Users/M2-Winterfell/Documents/Code/machine learning/FinalAssignment/database/data_before_pandemic_2.csv'
df.to_csv(output_file, index=False)

print(f"Updated data saved to {output_file}")
