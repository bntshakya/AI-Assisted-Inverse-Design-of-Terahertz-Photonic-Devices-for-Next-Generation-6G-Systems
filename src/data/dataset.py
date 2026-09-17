import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ComsolDataset(Dataset):
    def __init__(self, pt_file_path):
        # Load the binary tensor file
        data = torch.load(pt_file_path)
        self.x = data['geometries']
        self.y = data['em_responses']
        
    def __len__(self):
        # Tell PyTorch the total number of rows
        return len(self.x)
    
    def __getitem__(self, idx):
        # Fetch a single row (input and target) by its index
        return self.x[idx], self.y[idx]

def get_dataloaders(pt_file_path, batch_size=64, train_split=0.7, val_split=0.15):
    full_dataset = ComsolDataset(pt_file_path)
    total_size = len(full_dataset)
    
    # Calculate exact row counts for the 70/15/15 split
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create the DataLoaders (the conveyor belts)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# --- QUICK TEST ---
# This block only runs if you execute this specific file directly
if __name__ == "__main__":
    print("Testing the DataLoader pipeline...")
    
    # Point it to your newly compressed file
    file_path = "../../data/comsol/op.pt" # Adjust relative path if running from root
    
    # Generate the pipelines
    train, val, test = get_dataloaders('data/comsol/op.pt', batch_size=64)
    
    print(f"Total training batches: {len(train)}")
    
    # Grab exactly one batch off the conveyor belt to inspect it
    geom_batch, em_batch = next(iter(train))
    
    print("\nBatch Shapes:")
    print(f"Geometries: {geom_batch.shape} -> Expected: [64, 5]")
    print(f"EM Responses: {em_batch.shape} -> Expected: [64, Number of EM features]")