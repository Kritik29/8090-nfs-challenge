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


print(X.mean(axis=0))
print(X.std(axis=0))
