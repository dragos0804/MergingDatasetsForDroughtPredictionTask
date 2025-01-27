import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from datetime import datetime

def spatial_cluster_imputation(df, eps=1.0, min_samples=5):
    """
    Fill missing values using spatial clustering and hierarchical filling.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataset with spatial coordinates and features
    eps : float
        Maximum distance between two samples for DBSCAN clustering
    min_samples : int
        Minimum number of samples in a cluster for DBSCAN
        
    Returns:
    --------
    pandas DataFrame
        Dataset with imputed values
    """
    # Create a copy of the input DataFrame
    df_filled = df.copy()
    
    # Get coordinates for clustering
    coords = df[['LAT', 'LON']].values
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df_filled['cluster'] = clustering.labels_
    
    # Get list of columns to fill (excluding coordinate and cluster columns)
    columns_to_fill = df.columns.difference(['LAT', 'LON', 'cluster', 
                                           'BOTTOM_LEFT_LAT', 'BOTTOM_LEFT_LON',
                                           'UPPER_RIGHT_LAT', 'UPPER_RIGHT_LON'])
    
    # First pass: Fill within clusters
    for cluster_id in sorted(set(clustering.labels_)):
        if cluster_id == -1:  # Skip noise points
            continue
            
        cluster_mask = df_filled['cluster'] == cluster_id
        cluster_data = df_filled[cluster_mask]
        
        for column in columns_to_fill:
            # Calculate mean of non-NaN values in the cluster
            cluster_mean = cluster_data[column].mean()
            # Fill NaN values in the cluster with the cluster mean
            df_filled.loc[cluster_mask & df_filled[column].isna(), column] = cluster_mean
    
    # Second pass: Fill remaining NaNs using nearest clusters
    def fill_from_nearest_clusters(row, column):
        if pd.isna(row[column]):
            # Get distances to all other points
            point_coords = row[['LAT', 'LON']].values.reshape(1, -1)
            other_coords = df_filled[~df_filled[column].isna()][['LAT', 'LON']].values
            
            if len(other_coords) > 0:
                distances = cdist(point_coords, other_coords).flatten()
                # Get the nearest non-NaN value
                nearest_idx = distances.argmin()
                return df_filled[~df_filled[column].isna()][column].iloc[nearest_idx]
        return row[column]
    
    # Iterate through remaining NaN values
    for column in columns_to_fill:
        if df_filled[column].isna().any():
            df_filled[column] = df_filled.apply(
                lambda row: fill_from_nearest_clusters(row, column), axis=1
            )
    
    return df_filled.drop('cluster', axis=1)

def validate_imputation(df_original, df_imputed):
    """
    Validate the imputation results
    
    Parameters:
    -----------
    df_original : pandas DataFrame
        Original dataset with missing values
    df_imputed : pandas DataFrame
        Dataset with imputed values
        
    Returns:
    --------
    dict
        Validation metrics
    """
    metrics = {
        'total_nan_before': df_original.isna().sum().sum(),
        'total_nan_after': df_imputed.isna().sum().sum(),
        'columns_with_nans_before': df_original.isna().sum()[df_original.isna().sum() > 0].to_dict(),
        'columns_with_nans_after': df_imputed.isna().sum()[df_imputed.isna().sum() > 0].to_dict()
    }
    return metrics

def process_and_save_data(input_file, output_file=None, eps=0.1, min_samples=3):
    """
    Process the input CSV file, perform imputation, and save results
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str, optional
        Path to output CSV file. If None, generates a timestamped filename
    eps : float
        Maximum distance between points for DBSCAN clustering
    min_samples : int
        Minimum number of samples in a cluster
    """
    # Read input data
    df = pd.read_csv(input_file)
    
    # Perform imputation
    df_filled = spatial_cluster_imputation(df, eps=eps, min_samples=min_samples)
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'filled_data_{timestamp}.csv'
    
    # Save filled data
    df_filled.to_csv(output_file, index=False)
    
    # Get validation metrics
    metrics = validate_imputation(df, df_filled)
    
    # Save validation metrics
    metrics_file = output_file.rsplit('.', 1)[0] + '_validation.txt'
    with open(metrics_file, 'w') as f:
        f.write('Imputation Validation Results\n')
        f.write('===========================\n\n')
        f.write(f'Total NaN values before: {metrics["total_nan_before"]}\n')
        f.write(f'Total NaN values after: {metrics["total_nan_after"]}\n\n')
        f.write('Columns with NaN values before:\n')
        for col, count in metrics['columns_with_nans_before'].items():
            f.write(f'  {col}: {count}\n')
        f.write('\nColumns with NaN values after:\n')
        for col, count in metrics['columns_with_nans_after'].items():
            f.write(f'  {col}: {count}\n')
    
    return output_file, metrics_file

# Usage example
if __name__ == '__main__':
    input_file = '../output/FINAL_SOTWIS_NASA.csv'
    output_file = '../output/AUGUMENTED_SOTWIS_NASA.csv'
    
    filled_file, validation_file = process_and_save_data(
        input_file,
        output_file,
        eps=0.5,
        min_samples=5
    )
    
    print(f'Filled data saved to: {filled_file}')
    print(f'Validation results saved to: {validation_file}')