import os
import pandas as pd
from data.option_chain_cleaning import OptionChainCleaner

class CleanedOptionChain:
    def __init__(self, folder_path, spot, rate):
        """
        Initializes the processor with the directory path and market parameters.
        """
        self.folder_path = folder_path
        self.spot = spot
        self.rate = rate
        # Instantiate the cleaner once to be used across all files
        self.cleaner = OptionChainCleaner()

    def process_all(self):
        """
        Iterates through the target directory, cleans all CSVs, 
        and concatenates them into a single master DataFrame.
        """
        cleaned_dataframes = []
        
        # Safety check: Ensure the directory exists
        if not os.path.exists(self.folder_path):
            print(f"Directory not found: {self.folder_path}")
            return None

        # Iterate through the folder
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.folder_path, filename)
                
                try:
                    # Clean the individual file
                    df_cleaned = self.cleaner.clean_option_chain(file_path, self.spot, self.rate)
                    cleaned_dataframes.append(df_cleaned)
                    print(f"Processed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        # Concatenate and return
        if cleaned_dataframes:
            master_df = pd.concat(cleaned_dataframes, ignore_index=True)
            print(f"\nSuccessfully concatenated {len(cleaned_dataframes)} files!")
            return master_df
        else:
            print("No CSV files were found or successfully processed.")
            return None