import pandas as pd
import glob

def process_stroke_files():
    """
    Process stroke CSV files and combine them into a single DataFrame.
    
    Returns:
        pd.DataFrame: Combined DataFrame with columns [label, sample, x, y, z]
    """
    # Initialize empty list to store all dataframes
    all_data = []
    
    # Process files for each class (1-9)
    for class_num in range(1, 10):
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
            
            # Reorder columns and rename xyz coordinates
            df = df.rename(columns={0: 'x', 1: 'y', 2: 'z'})
            df = df[['label', 'sample', 'x', 'y', 'z']]
            
            # Append to our list of dataframes
            all_data.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df.sort_values(by=['label', 'sample'])

def main():
    # Create the combined DataFrame
    combined_df = process_stroke_files()

    # Display info about the combined DataFrame
    print("Combined DataFrame Info:")
    print(combined_df.info())
    print("\nFirst few rows:")
    print(combined_df.head())

    # Optionally save to CSV
    combined_df.to_csv('training_data/combined_strokes.csv', index=False)


if __name__ == "__main__":
    main()