import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def process_stroke_files():
    """
    Process stroke CSV files and combine them into a single DataFrame.
    
    Returns:
        pd.DataFrame: Combined DataFrame with columns [label, sample, time_step, x, y, z]
    """
    # Initialize empty list to store all dataframes
    all_data = []
    
    # Process files for each class (0-9)
    for class_num in range(0, 10):
        # Create pattern to match files for current class
        pattern = f'training_data/stroke_{class_num}_*.csv'
        
        # Get all matching files
        files = glob.glob(pattern)
        
        # Process each file for current class
        for file in files:
            # Extract sample number from filename
            sample_num = int(file.split('_')[-1].replace('.csv', ''))
            
            # Read the CSV file
            df = pd.read_csv(file, header=None)
            
            # Add label and sample columns
            df['label'] = class_num
            df['sample'] = sample_num
            
            # Add time_step column (0-based index for each point in the stroke)
            df['time_step'] = range(len(df))
            
            # Reorder columns and rename xyz coordinates
            df = df.rename(columns={0: 'x', 1: 'y', 2: 'z'})
            df = df[['label', 'sample', 'time_step', 'x', 'y', 'z']]
            
            # Append to our list of dataframes
            all_data.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True).sort_values(by=['label', 'sample', 'time_step'])
    
    return combined_df

def reduce_dimensions(df):
    """
    Perform dimension reduction on stroke data using PCA.
    
    Args:
        df (pd.DataFrame): Input DataFrame with columns [label, sample, time_step, x, y, z]
        
    Returns:
        pd.DataFrame: DataFrame with x, y, z replaced by 2 principal components
    """
    result_df = df.copy()
    
    # Group by label and sample to process each stroke separately
    groups = result_df.groupby(['label', 'sample'])
    
    # Initialize lists to store transformed data
    transformed_data = []
    
    for (label, sample), group in groups:
        # Extract coordinates
        coords = group[['x', 'y', 'z']].values
        
        # Normalize the coordinates
        scaler = StandardScaler()
        coords_normalized = scaler.fit_transform(coords)
        
        # Apply PCA
        pca = PCA(n_components=2)
        coords_transformed = pca.fit_transform(coords_normalized)
        
        # Create new dataframe with transformed coordinates
        transformed_group = pd.DataFrame({
            'label': label,
            'sample': sample,
            'time_step': group['time_step'],
            'pc1': coords_transformed[:, 0],
            'pc2': coords_transformed[:, 1]
        })
        
        transformed_data.append(transformed_group)
    
    # Combine all transformed groups
    result_df = pd.concat(transformed_data, ignore_index=True)
    
    # Sort to maintain order
    return result_df.sort_values(by=['label', 'sample', 'time_step'])

def main():
    # Create the combined DataFrame
    combined_df = process_stroke_files()

    # Display info about the combined DataFrame
    #print("Combined DataFrame Info:")
    #print(combined_df.info())
    #print("\nFirst few rows:")
    #print(combined_df.head())

    # Save to CSV
    combined_df.to_csv('training_data/combined_strokes.csv', index=False)

    df = pd.read_csv('training_data/combined_strokes.csv')

    # Apply dimension reduction
    reduced_df = reduce_dimensions(df)

    # Save the reduced data
    reduced_df.to_csv('training_data/reduced_strokes.csv', index=False)

    reduced_df = pd.read_csv('training_data/reduced_strokes.csv')


if __name__ == "__main__":
    main()