import geopandas as gpd
import pandas as pd
import os

def get_project_root():
    """
    Get the root directory of the project
    
    Returns:
    - Path to the project root directory
    """
    # Assume the script is in the 'src' folder
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def explore_soter_files():
    """
    Explore and list all files in the SOTER dataset folder
    
    Returns:
    - Dictionary of file types and their names
    """
    # Construct the path to the SOTER ShapeFiles folder
    soter_folder = os.path.join(get_project_root(), 'data', 'SOTWIS_SOVEUR_ver1.0', 'ShapeFiles')
    
    files = os.listdir(soter_folder)
    
    file_types = {
        'shp': [f for f in files if f.endswith('.shp')],
        'dbf': [f for f in files if f.endswith('.dbf')],
        'prj': [f for f in files if f.endswith('.prj')],
        'sbn': [f for f in files if f.endswith('.sbn')],
        'sbx': [f for f in files if f.endswith('.sbx')],
        'shx': [f for f in files if f.endswith('.shx')],
        'subfolders': [f for f in os.listdir(soter_folder) if os.path.isdir(os.path.join(soter_folder, f))]
    }
    
    return file_types, soter_folder

def convert_soter_to_csv(shapefile_name=None):
    """
    Convert SOTER shapefiles to CSV with geographic coordinates and soil parameters
    
    Parameters:
    - shapefile_name: Optional specific shapefile name to convert
                      If None, converts all .shp files in the SOTER folder
    
    Returns:
    - List of Pandas DataFrames with converted data
    """
    try:
        # Get the SOTER folder path
        file_types, soter_folder = explore_soter_files()
        
        # Get all .shp files
        shp_files = file_types['shp']
        if not shp_files:
            print("No shapefiles found!")
            return None
        
        # If a specific shapefile is provided, ensure it's in the list
        if shapefile_name:
            if shapefile_name not in shp_files:
                print(f"Shapefile {shapefile_name} not found in the directory.")
                return None
            shp_files = [shapefile_name]
        
        # Output folder for CSVs
        output_folder = os.path.join(get_project_root(), 'data', 'processed')
        os.makedirs(output_folder, exist_ok=True)
        
        # List to store converted DataFrames
        converted_dfs = []
        
        # Convert each shapefile
        for shp_file in shp_files:
            # Construct full path to the shapefile
            shapefile_path = os.path.join(soter_folder, shp_file)
            
            # Construct output CSV path
            output_csv_path = os.path.join(output_folder, shp_file.replace('.shp', '.csv'))
            
            # Read the shapefile
            gdf = gpd.read_file(shapefile_path)
            
            # Extract longitude and latitude from the geometry
            gdf['longitude'] = gdf.geometry.centroid.x
            gdf['latitude'] = gdf.geometry.centroid.y
            
            # Convert to DataFrame and drop geometry column
            df = gdf.drop(columns=['geometry'])
            
            # Save to CSV
            df.to_csv(output_csv_path, index=False)
            print(f"Conversion complete for {shp_file}. CSV saved to {output_csv_path}")
            print(f"Columns in the dataset: {list(df.columns)}")
            
            # Add to list of converted DataFrames
            converted_dfs.append(df)
        
        return converted_dfs
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    """
    Convert SOTER shapefile to CSV with geographic coordinates and soil parameters
    
    Parameters:
    - shapefile_name: Optional specific shapefile name to convert
    
    Returns:
    - Pandas DataFrame with converted data
    """
    try:
        # Get the SOTER folder path
        file_types, soter_folder = explore_soter_files()
        
        # If no specific shapefile is provided, use the first .shp file found
        if not shapefile_name:
            shp_files = file_types['shp']
            if not shp_files:
                print("No shapefile found!")
                return None
            shapefile_name = shp_files[0]
        
        # Construct full path to the shapefile
        shapefile_path = os.path.join(soter_folder, shapefile_name)
        
        # Construct output CSV path
        output_folder = os.path.join(get_project_root(), 'data', 'processed')
        os.makedirs(output_folder, exist_ok=True)
        output_csv_path = os.path.join(output_folder, shapefile_name.replace('.shp', '.csv'))
        
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Extract longitude and latitude from the geometry
        gdf['longitude'] = gdf.geometry.centroid.x
        gdf['latitude'] = gdf.geometry.centroid.y
        
        # Convert to DataFrame and drop geometry column
        df = gdf.drop(columns=['geometry'])
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"Conversion complete. CSV saved to {output_csv_path}")
        
        print(f"Columns in the dataset: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def inspect_shapefile_attributes(shapefile_name=None):
    """
    Inspect and print detailed information about the shapefile
    
    Parameters:
    - shapefile_name: Optional specific shapefile name to inspect
    """
    try:
        # Get the SOTER folder path
        file_types, soter_folder = explore_soter_files()
        
        # If no specific shapefile is provided, use the first .shp file found
        if not shapefile_name:
            shp_files = file_types['shp']
            if not shp_files:
                print("No shapefile found!")
                return
            shapefile_name = shp_files[0]
        
        # Construct full path to the shapefile
        shapefile_path = os.path.join(soter_folder, shapefile_name)
        
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)
        
        print("Shapefile Information:")
        print(f"Filename: {shapefile_name}")
        print(f"Total number of features: {len(gdf)}")
        print("\nAttribute Columns:")
        for column in gdf.columns:
            print(f"- {column}")
        
        print("\nSample Data (first 5 rows):")
        print(gdf.head())
        
        print("\nGeographic Coordinate System:")
        print(gdf.crs)
    
    except Exception as e:
        print(f"An error occurred while inspecting the shapefile: {e}")

# Example usage in the script
if __name__ == "__main__":
    # List all files in the SOTER folder
    files, folder = explore_soter_files()
    print("Shapefile(s) found:", files['shp'])
    
    # Optionally inspect a specific shapefile or the first one found
    inspect_shapefile_attributes()
    
    # Convert shapefile to CSV
    convert_soter_to_csv()