import pandas as pd
import torch
import os

def convert_and_split_csv(input_csv, output_dir='data/comsol', chunksize=100000, skiprows=9):
    os.makedirs(output_dir, exist_ok=True)
    geom_list = []
    em_list = []
    
    # Explicitly map the 7 columns in your COMSOL CSV
    comsol_columns = ['X', 'Y', 'L1', 'L2', 'k', 'lambda', 'freq']
    
    print(f"Processing {input_csv} in chunks...")
    
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, skiprows=skiprows, names=comsol_columns):   
        chunk = chunk.apply(pd.to_numeric, errors='coerce').dropna()
        
        # Extract safely by name
        geometries = chunk[['L1', 'L2']].values
        
        # Extract exactly 1 target feature by name
        em_responses = chunk[['freq']].values
        
        geom_list.append(torch.tensor(geometries, dtype=torch.float32))
        em_list.append(torch.tensor(em_responses, dtype=torch.float32))

    all_geometries = torch.cat(geom_list, dim=0)
    all_em_responses = torch.cat(em_list, dim=0)
    
    total_samples = len(all_geometries)
    print(f"Total dataset size: {total_samples:,} rows")
    
    # Generate random permutation of indices
    perm = torch.randperm(total_samples)
    
    train_end = int(0.70 * total_samples)
    val_end = int(0.85 * total_samples)
    
    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]
    
    # Slice tensors
    train_geom, train_em = all_geometries[train_idx], all_em_responses[train_idx]
    val_geom, val_em = all_geometries[val_idx], all_em_responses[val_idx]
    test_geom, test_em = all_geometries[test_idx], all_em_responses[test_idx]
    
    # Calculate normalization statistics strictly from TRAIN split
    x_mean, x_std = train_geom.mean(dim=0), train_geom.std(dim=0)
    y_mean, y_std = train_em.mean(dim=0), train_em.std(dim=0)
    
    x_std[x_std == 0] = 1.0
    y_std[y_std == 0] = 1.0
    
    # Save training normalization stats separately
    torch.save({
        'x_mean': x_mean, 'x_std': x_std,
        'y_mean': y_mean, 'y_std': y_std
    }, os.path.join(output_dir, 'norm_stats.pt'))
    
    # Save splits
    torch.save({'geometries': train_geom, 'em_responses': train_em}, os.path.join(output_dir, 'train.pt'))
    torch.save({'geometries': val_geom, 'em_responses': val_em}, os.path.join(output_dir, 'val.pt'))
    torch.save({'geometries': test_geom, 'em_responses': test_em}, os.path.join(output_dir, 'test.pt'))
    
    print("Successfully saved train.pt, val.pt, test.pt, and norm_stats.pt!")

if __name__ == "__main__":
    convert_and_split_csv(
        'data/comsol/parameter_sweep_values (Varied L1, L2 and r = 2Lby 3sqrt(3)).csv',
        'data/comsol'
    )