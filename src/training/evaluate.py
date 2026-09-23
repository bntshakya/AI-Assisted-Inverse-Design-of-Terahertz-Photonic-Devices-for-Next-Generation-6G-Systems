import torch
import matplotlib.pyplot as plt
import sys
import os

# Add root folder to Python's search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import get_dataloaders
from src.models.forward_model import ForwardModel

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load validation data (batch size 4 for a clean 2x2 plot)
    _, val_loader, _ = get_dataloaders("data/comsol/op.pt", batch_size=4)
    geometries, real_em = next(iter(val_loader))
    
    # Load trained model
    model = ForwardModel().to(device)
    model.load_state_dict(torch.load("models/saved/forward_model.pth", weights_only=True))
    model.eval()
    
    with torch.no_grad():
        predicted_em = model(geometries.to(device)).cpu()
        
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Forward Model Predictions vs COMSOL Ground Truth", fontsize=14)
    
    for i in range(4):
        ax = axs[i//2, i%2]
        ax.plot(real_em[i].numpy(), label="True COMSOL", marker='o', linestyle='dashed', color='blue')
        ax.plot(predicted_em[i].numpy(), label="Predicted (AI)", marker='x', linestyle='solid', color='red')
        
        # Generic title - update when you check your CSV headers
        ax.set_title(f"Sample {i+1} (Params: {geometries[i][0]:.2f}, {geometries[i][1]:.2f}...)")
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()