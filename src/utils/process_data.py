import pandas as pd
import torch
import os

def convert_csv_to_tensor(input_csv, output_pt, chunksize=100000, skiprows=9):
    geom_list = []
    em_list = []
    
    print(f"Processing {input_csv} in chunks...")
    
    # Read the massive CSV in manageable chunks (100,000 rows at a time)
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, skiprows=skiprows):   
        # 1. Force all data to be numeric. Any text (like a row of units) becomes 'NaN'
        chunk = chunk.apply(pd.to_numeric, errors='coerce')
        
        # 2. Drop any rows that became 'NaN' (safely deleting text/unit rows)
        chunk = chunk.dropna()
        
        # 3. Extract ONLY L1 and L2 (indices 2 and 3). 
        # Columns 0, 1, and 4 (X, Y, k) are ignored.
        geometries = chunk.iloc[:, 2:4].values
        
        # 4. Extract EM responses (index 5 onwards: lambda, freq, etc.)
        em_responses = chunk.iloc[:, 5:].values
        
        # 5. Convert to float32 tensors
        geom_list.append(torch.tensor(geometries, dtype=torch.float32))
        em_list.append(torch.tensor(em_responses, dtype=torch.float32))

    # Concatenate all lists into single massive tensors
    all_geometries = torch.cat(geom_list, dim=0)
    all_em_responses = torch.cat(em_list, dim=0)
    
    print(f"Extracted Geometries shape (L1, L2): {all_geometries.shape}")
    print(f"Extracted EM Responses shape: {all_em_responses.shape}")
    
    # Save as a highly compressed binary PyTorch file
    torch.save({
        'geometries': all_geometries, 
        'em_responses': all_em_responses
    }, output_pt)
    
    print(f"Saved successfully to {output_pt}!")

if __name__ == "__main__":
    convert_csv_to_tensor(
        'data/comsol/parameter_sweep_values (Varied L1, L2 and r = 2Lby 3sqrt(3)).csv',
        'data/comsol/op.pt'
    )