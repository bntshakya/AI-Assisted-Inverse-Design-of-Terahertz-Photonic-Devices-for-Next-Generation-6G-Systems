import torch
import sys
import os

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.backward_model import BackwardModel
from src.models.forward_model import ForwardModel

def predict_geometry(target_f_lower, target_f_upper):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- INVERSE DESIGN INFERENCE ---")
    print(f"Target Bandgap: [{target_f_lower} GHz, {target_f_upper} GHz]\n")
    
    # 1. Load Normalization Stats
    stats = torch.load("data/comsol/norm_stats.pt", map_location=device, weights_only=True)
    x_mean, x_std = stats['x_mean'].to(device), stats['x_std'].to(device)
    y_mean, y_std = stats['y_mean'].to(device), stats['y_std'].to(device)
    
    # 2. Load Models
    backward_model = BackwardModel().to(device)
    backward_model.load_state_dict(torch.load("models/saved/backward_model.pth", map_location=device, weights_only=True))
    backward_model.eval()

    forward_model = ForwardModel().to(device)
    forward_model.load_state_dict(torch.load("models/saved/forward_model.pth", map_location=device, weights_only=True))
    forward_model.eval()
    
    # 3. Prepare the Input (2 features: f_lower, f_upper)
    target_tensor = torch.tensor([[target_f_lower, target_f_upper]], dtype=torch.float32).to(device)
    target_norm = (target_tensor - y_mean) / y_std
    
    # 4. Predict Geometry (Inverse) and Reconstruct Frequency (Forward)
    with torch.no_grad():
        pred_geom_norm = backward_model(target_norm)
        reconstructed_freq_norm = forward_model(pred_geom_norm)
        
    # 5. Un-normalize
    pred_geom = (pred_geom_norm * x_std) + x_mean
    reconstructed_freq = (reconstructed_freq_norm * y_std) + y_mean
    
    # Extract values
    pred_L1 = pred_geom[0, 0].item()
    pred_L2 = pred_geom[0, 1].item()
    recon_f_lower = reconstructed_freq[0, 0].item()
    recon_f_upper = reconstructed_freq[0, 1].item()
    
    print(">>> AI PREDICTED GEOMETRY <<<")
    print(f"L1: {pred_L1:.6f} um")
    print(f"L2: {pred_L2:.6f} um\n")
    
    print(">>> FORWARD MODEL RECONSTRUCTION CHECK <<<")
    print(f"If you build this geometry, the predicted bandgap is:")
    print(f"f_lower: {recon_f_lower:.2f} GHz  (Error: {abs(recon_f_lower - target_f_lower):.2f} GHz)")
    print(f"f_upper: {recon_f_upper:.2f} GHz  (Error: {abs(recon_f_upper - target_f_upper):.2f} GHz)")
    print("--------------------------------\n")

if __name__ == "__main__":
    print("\n=== Photonic Inverse Design Predictor ===")
    try:
        f_lower = float(input("Enter desired LOWER bandgap frequency (GHz): "))
        f_upper = float(input("Enter desired UPPER bandgap frequency (GHz): "))
        
        # Make sure they didn't mix up the numbers
        if f_lower >= f_upper:
            print("Error: The lower frequency must be smaller than the upper frequency!")
        else:
            predict_geometry(f_lower, f_upper)
            
    except ValueError:
        print("Error: Please enter valid numbers (e.g., 315.5).")