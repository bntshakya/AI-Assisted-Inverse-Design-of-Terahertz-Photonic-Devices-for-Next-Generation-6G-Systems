import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import ComsolDataset, get_dataloaders
from src.models.backward_model import BackwardModel

def main():
    pt_path = "data/comsol/op.pt"
    model_path = "models/saved/backward_model.pth"
    export_path = "data/comsol/exports/ai_predicted_params.txt"

    # 1. Load dataset object for exact stats
    dataset = ComsolDataset(pt_path)

    # 2. Pull 1 real validation sample (Ground Truth geometry and Target EM)
    _, val_loader, _ = get_dataloaders(pt_path, batch_size=1)
    real_geometries_norm, target_em_norm = next(iter(val_loader))

    # 3. Load trained Backward Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backward_model = BackwardModel().to(device)
    backward_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    backward_model.eval()

    # 4. Predict geometry from target EM
    with torch.no_grad():
        predicted_norm = backward_model(target_em_norm.to(device)).cpu().squeeze()

    # 5. Reverse normalization to physical units
    ai_physical = (predicted_norm * dataset.x_std) + dataset.x_mean
    gt_physical = (real_geometries_norm.squeeze() * dataset.x_std) + dataset.x_mean
    target_em_physical = (target_em_norm.squeeze() * dataset.y_std) + dataset.y_mean

    # 6. Display Target Frequency & Ground Truth Comparison
    print("\n========================================================")
    print(f" TARGET FREQUENCY : {target_em_physical[1].item():.4f} GHz")
    print(f" TARGET WAVELENGTH: {target_em_physical[0].item():.6e}")
    print("========================================================")
    print(f"{'Parameter':<10} | {'Ground Truth':<22} | {'AI Predicted':<22}")
    print("-" * 60)

    param_names = ['X', 'Y', 'L1', 'L2', 'k']
    units = ['[um]', '[um]', '', '', '']

    for name, gt, ai in zip(param_names, gt_physical, ai_physical):
        print(f"{name:<10} | {gt.item():<22.8e} | {ai.item():<22.8e}")
    
    # 7. Write parameters to file for COMSOL
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, 'w') as f:
        for name, val, unit in zip(param_names, ai_physical, units):
            f.write(f"{name} {val.item():.8e}{unit} AI_Predicted\n")

    print(f"\nFile written: {export_path}")

if __name__ == "__main__":
    main()