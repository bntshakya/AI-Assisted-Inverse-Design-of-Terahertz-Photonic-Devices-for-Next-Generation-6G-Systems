import torch
import matplotlib.pyplot as plt
import os
import sys

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import get_dataloaders
from src.models.forward_model import ForwardModel
from src.models.backward_model import BackwardModel

def evaluate_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 1. Load the test data and normalization statistics
    _, _, test_loader = get_dataloaders(data_dir="data/comsol", batch_size=100)
    stats = torch.load("data/comsol/norm_stats.pt", weights_only=True)
    
    x_mean, x_std = stats['x_mean'].to(device), stats['x_std'].to(device)
    y_mean, y_std = stats['y_mean'].to(device), stats['y_std'].to(device)

    # 2. Load the trained models
    forward_model = ForwardModel().to(device)
    forward_model.load_state_dict(torch.load("models/saved/forward_model.pth", map_location=device, weights_only=True))
    forward_model.eval()

    backward_model = BackwardModel().to(device)
    backward_model.load_state_dict(torch.load("models/saved/backward_model.pth", map_location=device, weights_only=True))
    backward_model.eval()

    # 3. Get one batch of test data
    true_geom_norm, true_em_norm = next(iter(test_loader))
    true_geom_norm = true_geom_norm.to(device)
    true_em_norm = true_em_norm.to(device)

    # 4. Run the Tandem Pipeline
    with torch.no_grad():
        # Inverse Design: Predict geometry from Target EM
        pred_geom_norm = backward_model(true_em_norm)
        # Reconstruct EM: Predict EM from the generated geometry
        reconstructed_em_norm = forward_model(pred_geom_norm)

    # 5. Un-normalize back to physical values
    true_geom = (true_geom_norm * x_std) + x_mean
    pred_geom = (pred_geom_norm * x_std) + x_mean
    
    true_em = (true_em_norm * y_std) + y_mean
    reconstructed_em = (reconstructed_em_norm * y_std) + y_mean

    # Convert to numpy for matplotlib
    true_geom = true_geom.cpu().numpy()
    pred_geom = pred_geom.cpu().numpy()
    true_em = true_em.cpu().numpy()
    reconstructed_em = reconstructed_em.cpu().numpy()

    # 6. Plotting
    plt.figure(figsize=(15, 6))

    # Plot A: True vs Predicted L1
    plt.subplot(1, 3, 1)
    plt.scatter(true_geom[:, 0], pred_geom[:, 0], alpha=0.5, color='blue')
    plt.plot([true_geom[:, 0].min(), true_geom[:, 0].max()], 
             [true_geom[:, 0].min(), true_geom[:, 0].max()], 'r--', lw=2)
    plt.xlabel("True L1 (um)")
    plt.ylabel("Predicted L1 (um)")
    plt.title("Inverse Design: L1 Accuracy")
    plt.grid(True)

    # Plot B: True vs Predicted L2
    plt.subplot(1, 3, 2)
    plt.scatter(true_geom[:, 1], pred_geom[:, 1], alpha=0.5, color='green')
    plt.plot([true_geom[:, 1].min(), true_geom[:, 1].max()], 
             [true_geom[:, 1].min(), true_geom[:, 1].max()], 'r--', lw=2)
    plt.xlabel("True L2 (um)")
    plt.ylabel("Predicted L2 (um)")
    plt.title("Inverse Design: L2 Accuracy")
    plt.grid(True)

    # Plot C: Target EM vs Reconstructed EM (Feature 1)
    plt.subplot(1, 3, 3)
    plt.scatter(true_em[:, 0], reconstructed_em[:, 0], alpha=0.5, color='purple')
    plt.plot([true_em[:, 0].min(), true_em[:, 0].max()], 
             [true_em[:, 0].min(), true_em[:, 0].max()], 'r--', lw=2)
    plt.xlabel("Target EM Feature 1")
    plt.ylabel("Reconstructed EM Feature 1")
    plt.title("Tandem Reconstructed EM")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("models/saved/tandem_evaluation.png", dpi=300)
    print("Evaluation complete! Plot saved to models/saved/tandem_evaluation.png")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()