import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import get_dataloaders
from src.models.forward_model import ForwardModel
from src.models.backward_model import BackwardModel

def evaluate_tandem():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load validation data
    _, val_loader, _ = get_dataloaders("data/comsol/op.pt", batch_size=4)
    real_geometries, target_em = next(iter(val_loader))
    
    # 2. Load both trained models
    forward_model = ForwardModel().to(device)
    forward_model.load_state_dict(torch.load("models/saved/forward_model.pth", weights_only=True))
    forward_model.eval()

    backward_model = BackwardModel().to(device)
    backward_model.load_state_dict(torch.load("models/saved/backward_model.pth", weights_only=True))
    backward_model.eval()

    # 3. Perform Inverse Design Inference
    with torch.no_grad():
        # Pass target EM through backward model -> Get predicted 5 geometric parameters
        predicted_geometries = backward_model(target_em.to(device))
        # Pass predicted parameters through forward model -> Get reconstructed EM
        reconstructed_em = forward_model(predicted_geometries).cpu()

    # 4. Print Predicted Unit Cell Parameters for COMSOL
    print("\n--- SAMPLE AI-GENERATED GEOMETRIC PARAMETERS ---")
    for i in range(4):
        params = predicted_geometries[i].cpu().numpy()
        print(f"Sample {i+1} Parameters (5-dim): [{params[0]:.4f}, {params[1]:.4f}, {params[2]:.4f}, {params[3]:.4f}, {params[4]:.4f}]")
    print("------------------------------------------------\n")

    # 5. Plot Target EM vs Reconstructed EM
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Tandem Inverse Design: Target EM vs Reconstructed EM", fontsize=14)
    
    for i in range(4):
        ax = axs[i//2, i%2]
        ax.plot(target_em[i].numpy(), label="Target EM", marker='o', linestyle='dashed', color='green')
        ax.plot(reconstructed_em[i].numpy(), label="Reconstructed (AI)", marker='x', linestyle='solid', color='purple')
        ax.set_title(f"Target Sample {i+1}")
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_tandem()