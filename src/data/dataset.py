import os
import torch
from torch.utils.data import Dataset, DataLoader

class ComsolDataset(Dataset):
    def __init__(self, split_pt_path, stats_pt_path='data/comsol/norm_stats.pt'):
        data = torch.load(split_pt_path)
        stats = torch.load(stats_pt_path)
        
        raw_x = data['geometries'].float()
        raw_y = data['em_responses'].float()
        
        # Always normalize using TRAIN stats
        self.x = (raw_x - stats['x_mean']) / stats['x_std']
        self.y = (raw_y - stats['y_mean']) / stats['y_std']
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloaders(data_dir='data/comsol', batch_size=64):
    train_ds = ComsolDataset(f"{data_dir}/train.pt", f"{data_dir}/norm_stats.pt")
    val_ds   = ComsolDataset(f"{data_dir}/val.pt",   f"{data_dir}/norm_stats.pt")
    test_ds  = ComsolDataset(f"{data_dir}/test.pt",  f"{data_dir}/norm_stats.pt")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader