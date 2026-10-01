import pandas as pd
import torch
import os

def extract_bandgap_features(input_csv, output_dir='data/comsol'):
    """
    Reads the COMSOL mesh export, drops redundant spatial nodes (X, Y),
    and extracts the absolute bandgap edges for each (L1, L2) configuration
    by dynamically scanning across all k-points.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {input_csv} in chunks to extract bandgap edges...")
    
    chunksize = 1_000_000
    unique_rows = []
    
    # Columns: % X, Y, L1, L2, k, lambda, emw.freq (GHz)
    # We usecols 2 (L1), 3 (L2), 4 (k), 6 (freq)
    for i, chunk in enumerate(pd.read_csv(input_csv, skiprows=9, header=None, 
                                          usecols=[2, 3, 4, 6], 
                                          names=['L1', 'L2', 'k', 'freq'],
                                          chunksize=chunksize)):
        
        # 1. Force freq to numeric, drop complex errors safely
        chunk['freq'] = pd.to_numeric(chunk['freq'], errors='coerce')
        chunk = chunk.dropna()
        
        # 2. Round k to avoid floating point issues
        chunk['k'] = chunk['k'].round(3)
        
        # 3. Drop all the duplicate (X, Y) nodes instantly!
        chunk = chunk.drop_duplicates()
        
        unique_rows.append(chunk)
        print(f"Processed chunk {i+1}...")

    # Combine all chunks and do one final drop_duplicates
    df_clean = pd.concat(unique_rows).drop_duplicates()
    print(f"\nReduced dataset to {len(df_clean)} unique physics states (from ~29.5M rows)!")

    # Now, to find the true bandgap across ALL k for a given (L1, L2):
    print("Dynamically scanning band structures to find absolute bandgaps...")
    
    # For a specific L1, L2, k, there should be 2 frequencies (Lower band & Upper band)
    # We group by L1, L2, k to get min (lower band) and max (upper band) at each k
    bands = df_clean.groupby(['L1', 'L2', 'k'])['freq'].agg(['min', 'max']).reset_index()
    bands.rename(columns={'min': 'lower_band_freq', 'max': 'upper_band_freq'}, inplace=True)
    
    # Finally, for each L1, L2 over all k:
    # Lower band edge = absolute maximum of the lower band over all k
    # Upper band edge = absolute minimum of the upper band over all k
    final_gaps = bands.groupby(['L1', 'L2']).agg(
        f_lower=('lower_band_freq', 'max'),
        f_upper=('upper_band_freq', 'min')
    ).reset_index()
    
    # Filter out geometries that don't have a valid bandgap (e.g. bands crossed completely)
    final_df = final_gaps[final_gaps['f_lower'] < final_gaps['f_upper']].copy()
    
    total_samples = len(final_df)
    print(f"Final usable dataset size: {total_samples} unique (L1, L2) configurations.")
    
    # Convert to PyTorch tensors
    all_geometries = torch.tensor(final_df[['L1', 'L2']].values, dtype=torch.float32)
    all_em_responses = torch.tensor(final_df[['f_lower', 'f_upper']].values, dtype=torch.float32)
    
    # Generate random permutation for Train (70%), Val (15%), Test (15%) splits
    perm = torch.randperm(total_samples)
    train_end = int(0.70 * total_samples)
    val_end = int(0.85 * total_samples)
    
    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]
    
    train_geom, train_em = all_geometries[train_idx], all_em_responses[train_idx]
    val_geom, val_em = all_geometries[val_idx], all_em_responses[val_idx]
    test_geom, test_em = all_geometries[test_idx], all_em_responses[test_idx]
    
    # Calculate normalization statistics strictly from TRAIN split
    x_mean, x_std = train_geom.mean(dim=0), train_geom.std(dim=0)
    y_mean, y_std = train_em.mean(dim=0), train_em.std(dim=0)
    
    x_std[x_std == 0] = 1.0
    y_std[y_std == 0] = 1.0
    
    # Save training normalization stats
    torch.save({
        'x_mean': x_mean, 'x_std': x_std,
        'y_mean': y_mean, 'y_std': y_std
    }, os.path.join(output_dir, 'norm_stats.pt'))
    
    # Save splits
    torch.save({'geometries': train_geom, 'em_responses': train_em}, os.path.join(output_dir, 'train.pt'))
    torch.save({'geometries': val_geom, 'em_responses': val_em}, os.path.join(output_dir, 'val.pt'))
    torch.save({'geometries': test_geom, 'em_responses': test_em}, os.path.join(output_dir, 'test.pt'))
    
    print("Successfully processed and saved train.pt, val.pt, test.pt, and norm_stats.pt!")

if __name__ == "__main__":
    csv_file = 'data/comsol/parameter_sweep_values (Varied L1, L2 and r = 2Lby 3sqrt(3)).csv'
    if os.path.exists(csv_file):
        extract_bandgap_features(csv_file, 'data/comsol')
    else:
        print(f"Could not find {csv_file}")
