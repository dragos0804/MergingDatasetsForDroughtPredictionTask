import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Function to resolve conflicts within a group
def resolve_conflicts_fast(group):
    data = group.to_numpy()
    columns = group.columns

    n_rows, n_cols = data.shape
    merged_mask = np.zeros((n_rows, n_rows), dtype=bool)

    results = []
    for i in range(n_rows):
        if merged_mask[i].any():
            continue  # Skip rows already merged

        merged_row = data[i].copy()
        conflict_found = False

        for j in range(i + 1, n_rows):
            row_j = data[j]
            overlap = ~pd.isna(merged_row) & ~pd.isna(row_j)
            if np.any(overlap & (merged_row != row_j)):
                conflict_found = True
                continue

            merged_row = np.where(pd.isna(merged_row), row_j, merged_row)
            merged_mask[i, j] = True
            merged_mask[j, i] = True

        results.append(merged_row if not conflict_found else data[i])

    return pd.DataFrame(results, columns=columns)

# Function to apply parallel processing
def merge_duplicates_parallel(input_file, output_file, n_jobs=-1):
    df = pd.read_csv(input_file)

    # Group by the unique identifier fields
    grouped = df.groupby(['LAT', 'LON', 'YEAR'])

    # Resolve conflicts in parallel
    resolved_groups = Parallel(n_jobs=n_jobs)(
        delayed(resolve_conflicts_fast)(group) for _, group in grouped
    )

    # Combine all resolved groups into a single DataFrame
    resolved_df = pd.concat(resolved_groups, ignore_index=True)

    # Save the resolved DataFrame
    resolved_df.to_csv(output_file, index=False)
    print(f"Resolved dataset saved to {output_file}")

# Example usage
merged_file = '../output/merged_sotwis_nasa.csv'  # Input file
output_file = '../output/resolved_sotwis_nasa.csv'  # Output file

merge_duplicates_parallel(merged_file, output_file)

def drop_pattern_columns(input_csv, output_csv, column_patterns, specific_columns):
    """
    Drop columns matching patterns or specific names from a CSV file and save the result to a new file.

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the modified CSV file.
        column_patterns (list): List of patterns to match column names (e.g., 'SOIL', 'PROP').
        specific_columns (list): List of specific column names to drop (e.g., ['PARAMETER']).
    """
    try:
        # Load the CSV into a DataFrame
        df = pd.read_csv(input_csv)
        
        # Find columns matching the patterns
        columns_to_drop = [
            col for col in df.columns
            if any(col.startswith(pattern) for pattern in column_patterns)
        ]
        
        # Add the specific columns to the list
        columns_to_drop.extend(specific_columns)
        
        # Drop the identified columns
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Save the modified DataFrame to a new CSV file
        df.to_csv(output_csv, index=False)
        print(f"Columns {columns_to_drop} dropped and saved to {output_csv}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
input_csv = '../output/resolved_sotwis_nasa.csv'
output_csv = '../output/FINAL_SOTWIS_NASA.csv'
column_patterns = ['SOIL', 'PROP']  # Pattern prefixes for column names
specific_columns = ['PARAMETER']    # Explicit column name to drop

drop_pattern_columns(input_csv, output_csv, column_patterns, specific_columns)

