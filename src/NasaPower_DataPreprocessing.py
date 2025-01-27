import os
import pandas as pd

# Define the input and output directories
input_dir = "../data/NasaPower"
output_file = "../output/merged_nasa_power.csv"

# Initialize an empty DataFrame to store merged data
merged_df = pd.DataFrame()

# Loop through all CSV files in the input directory
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(input_dir, file)

        try:
            # Read the CSV file, skipping the header lines (assumed to be the first 9 rows)
            temp_df = pd.read_csv(file_path, skiprows=9)

            # Append the data to the merged DataFrame
            merged_df = pd.concat([merged_df, temp_df], ignore_index=True)
        except pd.errors.ParserError as e:
            print(f"Error reading {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error with {file_path}: {e}")

# Drop duplicate rows based on all columns
merged_df = merged_df.drop_duplicates()

# Save the merged DataFrame to the output file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
merged_df.to_csv(output_file, index=False)

print(f"Merged CSV saved to {output_file}")

