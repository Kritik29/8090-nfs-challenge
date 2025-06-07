import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# === Load and flatten data ===
with open('public_cases.json', 'r') as f:
    raw_data = json.load(f)

records = []
for case in raw_data:
    row = case['input']
    row['reimbursement'] = case['expected_output']
    records.append(row)

df = pd.DataFrame(records)

X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
y = df['reimbursement'].values.reshape(-1, 1)

# === Log transform target to reduce skew ===
y_log = np.log1p(y)  # log(1 + y) to handle zero values safely

# === Normalize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-validation-test split (80/10/10) ===
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y_log, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)  # 0.1111 * 0.9 ~= 0.1

# === Convert to PyTorch tensors ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# === Dataset and DataLoader ===
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === Define the neural network with dropout ===
class ReimbursementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ReimbursementNet()

# === Loss and optimizer ===
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# === Early stopping params ===
best_val_loss = float('inf')
patience = 100
trigger_times = 0

EPOCHS = 5000
for epoch in range(EPOCHS):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_preds = model(val_X)
            val_loss = criterion(val_preds, val_y)
            val_losses.append(val_loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pth')  # save best model
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} — Validation MSE: {avg_val_loss:.4f}")

# === Load best model ===
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# === Evaluate on test set ===
with torch.no_grad():
    y_test_pred_log = model(X_test_tensor).numpy()
    # Inverse transform from log scale
    y_test_pred = np.expm1(y_test_pred_log)
    y_test_orig = np.expm1(y_test)

    mse = mean_squared_error(y_test_orig, y_test_pred)
    rmse = np.sqrt(mse)
    print(f"\nFinal Test RMSE (original scale): {rmse:.2f}")
