import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

def merge_datasets_efficiently(nasa_file, sotwis_file, output_file):
    # Load the NASA dataset
    nasa_df = pd.read_csv(nasa_file)
    nasa_gdf = gpd.GeoDataFrame(
        nasa_df,
        geometry=gpd.points_from_xy(nasa_df['LON'], nasa_df['LAT'])
    )

    # Load the SOTWIS dataset and convert bounding boxes to polygons
    sotwis_df = pd.read_csv(sotwis_file)
    sotwis_gdf = gpd.GeoDataFrame(
        sotwis_df,
        geometry=[
            Polygon([
                (row['BOTTOM_LEFT_LON'], row['BOTTOM_LEFT_LAT']),
                (row['BOTTOM_LEFT_LON'], row['UPPER_RIGHT_LAT']),
                (row['UPPER_RIGHT_LON'], row['UPPER_RIGHT_LAT']),
                (row['UPPER_RIGHT_LON'], row['BOTTOM_LEFT_LAT']),
                (row['BOTTOM_LEFT_LON'], row['BOTTOM_LEFT_LAT'])
            ])
            for _, row in sotwis_df.iterrows()
        ]
    )

    # Ensure coordinate systems match
    nasa_gdf.set_crs("EPSG:4326", inplace=True)
    sotwis_gdf.set_crs("EPSG:4326", inplace=True)

    # Perform a spatial join to match NASA points to SOTWIS polygons
    merged_gdf = gpd.sjoin(nasa_gdf, sotwis_gdf, how='inner', predicate='within')
    if 'index_right' in merged_gdf.columns:
        merged_gdf.drop(columns=['index_right'], inplace=True)

    # Drop geometry columns and save the result
    merged_df = pd.DataFrame(merged_gdf.drop(columns='geometry'))
    merged_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")

# Example usage
nasa_file = '../output/merged_nasa_power.csv'  # NASA Power Data
sotwis_file = '../output/sotwis_processed.csv'  # SOTWIS Processed Data
output_file = '../output/merged_sotwis_nasa.csv'  # Output merged data

merge_datasets_efficiently(nasa_file, sotwis_file, output_file)
