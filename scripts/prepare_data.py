#prepare the input datasets for subsequent neural network training and evaluation
#You start with CSV files containing information about particles and their reconstruction in two different simulation approaches:
#- MC Overlay: The conventional overlay simulation.
#- Track Overlay: The alternative, faster simulation.
#The script initiates the processing of these files and produces datasets that can later be used to train a classifier.
#Data preparation: Construct the physics quantities and target labels, and save them to HDF5 files.
#Feature selection: Decide which columns of the processed data are used as neural network inputs.

#Usage:
#singularity exec dsnnr_4gpu_v5.sif python scripts/prepare_data.py --sample JZ7W --path data
#many other arguments are possible, see args.py

#sys: information about the Python runtime and interpreter.
import sys

#os: functions for interacting with the operating system, including file paths and directories.
import os

# Add parent directory to path so we can import from data/, network/, utils/
#os.path.dirname(__file__) returns the path where this script is located
#os.path.join(os.path.dirname(__file__), '..') is the parent directory
#Python needs to know where the data and utils packages are located.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle      #to save metadata as a serialized Python object
import datetime    #Record when data preparation took place

from data.dataloader_multi import prepareData
from utils.args import get_args

def main():
    
    #This retrieves the configuration options
    args = get_args()
    #usage, for example: args.sample
    
    #record the current date and time
    datetimenow = datetime.datetime.now().strftime("%B %d, %Y: %H:%M:%S")
    
    # Settings-------
    
    #This initializes NumPy's random number generator with a fixed seed.
    np.random.seed(args.seed)
    
    #This stores the intended training fraction.
    split = args.trainsplit
    
    use_density = bool(args.usedensity)
    sample = args.sample
    path = args.path
    
    # Output directory for processed data
    processed_dir = f"data/processed/{sample}"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create log file
    log_file = open(f"{processed_dir}/prepare_data_log.txt", "w")   #w means write
    print(f"Data preparation started: {datetimenow}", file=log_file, flush=True)
    print("="*60, file=log_file, flush=True)
    
    # Load and process data
    print(f"Loading data from {path}/{sample}.h5", file=log_file, flush=True)
    print(f"Train/test split: {split}/{1-split}", file=log_file, flush=True)
    print(f"Use density features: {use_density}", file=log_file, flush=True)
    
    # Prepare the raw data (load CSVs, compute features, save to HDF5)
    #prepareData() is defined in dataloader_multi.py
    #That function reads the MC Overlay and Track Overlay CSV files, aligns their entries, constructs the combined dataset,
    #calculates additional variables, and defines the classification target. It then saves the processed data
    # it returns 3 objects: First two are Pandas DataFrames, and the third is a list of feature names.
    #fullData_pos: Rows with TargetLabel = 1. The two overlay approaches disagree about the matching outcome.
    #fullData_neg: Rows with TargetLabel = 0. The two overlay approaches agree.
    #feat_list: Names of the selected ML input features
    fullData_pos, fullData_neg, feat_list = prepareData(
        path=path,
        logfile=log_file,
        sample=sample,
        use_densities=use_density
    )
    
    print(
        f"\nData preparation complete:",
        file=log_file,
        flush=True,
    )
    
    #shape means dimensions, ie number of columns and rows
    print(f"  Matched samples (TargetLabel=1): {fullData_pos.shape}", file=log_file, flush=True)
    print(f"  Unmatched samples (TargetLabel=0): {fullData_neg.shape}", file=log_file, flush=True)
    print(f"  Number of features: {len(feat_list)}", file=log_file, flush=True)
    print(f"  Feature list: {feat_list}", file=log_file, flush=True)
    
    # Save metadata for training/evaluation scripts
    print(f"\nSaving metadata to {processed_dir}", file=log_file, flush=True)
    
    #This creates a Python dictionary. A dictionary stores information as key-value pairs.
    #example: metadata["sample"] will return sample
    #shape[0] returns the number of rows in the dataframe objects
    metadata = {
        "feat_list": feat_list,
        "num_features": len(feat_list),
        "sample": sample,
        "split": split,
        "use_density": use_density,
        "seed": args.seed,
        "preparation_date": datetimenow,
        "pos_samples": fullData_pos.shape[0],
        "neg_samples": fullData_neg.shape[0]
    }
    with open(f"{processed_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    #This above writes the metadata dictionary to a file. wb means write binary.
    
    print(f"\n✓ Data preparation complete!", file=log_file, flush=True)
    print(f"  - Processed data: {path}/matched_{sample}_data.h5, {path}/unmatched_{sample}_data.h5", file=log_file, flush=True)
    print(f"  - Metadata: {processed_dir}/metadata.pkl", file=log_file, flush=True)
    print("="*60, file=log_file, flush=True)
    
    log_file.close()
    
    # Also print to console
    print(f"✓ Data preparation complete!")
    print(f"  Matched samples (TargetLabel=1): {fullData_pos.shape[0]}")
    print(f"  Unmatched samples (TargetLabel=0): {fullData_neg.shape[0]}")
    print(f"  Features: {len(feat_list)}")
    print(f"  Metadata saved to: {processed_dir}/metadata.pkl")

#this actually starts the script This is a standard Python pattern.
#python scripts/prepare_data.py --sample ttbar causes main() to run
if __name__ == "__main__":
    main()
