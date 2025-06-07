import sys
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# Hardcoded scaler params from training data (replace with your actual means/stds)
mean = np.array([7.04300, 597.41374, 1211.05687])    
std = np.array([3.926139, 351.299790, 742.854180])      

class ReimbursementNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)

    # Parse inputs
    inputs = np.array([float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])])

    # Normalize inputs
    X_norm = (inputs - mean) / std
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)  # batch size 1

    # Load model
    model = ReimbursementNet()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Predict log1p output
    with torch.no_grad():
        y_log_pred = model(X_tensor).item()

    # Inverse transform
    reimbursement_pred = np.expm1(y_log_pred)

    # Print prediction only
    print(f"{reimbursement_pred:.2f}")

if __name__ == "__main__":
    main()
