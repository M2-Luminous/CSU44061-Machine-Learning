import pandas as pd

# Load the CSV file
df = pd.read_csv('C:/Users/M2-Winterfell/Documents/Code/machine learning/FinalAssignment/database/after_pandemic.csv')

# Delete the 'LAST UPDATED' column
df.drop('LAST UPDATED', axis=1, inplace=True)

# Convert the 'TIME' column to datetime
df['TIME'] = pd.to_datetime(df['TIME'])

# Create new 'DATE' column with desired format (replace "-" with "/")
df['DATE'] = df['TIME'].dt.strftime('%Y-%m-%d').str.replace('-0', '/').str.replace('-', '/').str.replace('/0', '/', 1)

# Extract time part
df['TIME'] = df['TIME'].dt.strftime('%H:%M:%S')

# Reorder columns if necessary
df = df[['TIME', 'DATE'] + [col for col in df.columns if col not in ['TIME', 'DATE']]]

# Save the modified DataFrame to a new CSV file
df.to_csv('C:/Users/M2-Winterfell/Documents/Code/machine learning/FinalAssignment/database/after_pandemic.csv', index=False)
