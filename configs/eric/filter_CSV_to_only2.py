import pandas as pd

# Read the input CSV file
input_file = '/Users/ewern/Downloads/DataCatSep26/Data.csv'
output_file = '/Users/ewern/Downloads/DataCatSep26/Data_only2.csv'

# Read the CSV file
df = pd.read_csv(input_file)

# Remove rows containing -1.0
df_filtered = df[~df.isin([-1.0]).any(axis=1)]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv(output_file, index=False)

print(f"Filtered CSV saved to: {output_file}")
print(f"Original row count: {len(df)}")
print(f"Filtered row count: {len(df_filtered)}")