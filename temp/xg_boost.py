import xgboost as xgb
import json
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === Load and flatten the data ===
with open('public_cases.json', 'r') as f:
    raw_data = json.load(f)

records = []
for case in raw_data:
    row = case['input']
    row['reimbursement'] = case['expected_output']
    records.append(row)

df = pd.DataFrame(records)

# === Features and target ===
X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
y = df['reimbursement']

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Objective function for Optuna ===
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # or 'gpu_hist' if using GPU
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

# === Run Optuna ===
study = optuna.create_study(direction='minimize')

MAX_TRIALS = 300
TARGET_MSE = 3000
total_trials_run = 0
batch_size = 25

while total_trials_run < MAX_TRIALS:
    study.optimize(objective, n_trials=batch_size)
    total_trials_run += batch_size

    current_best = study.best_value
    print(f"🔍 Trials: {total_trials_run} — Best MSE so far: {current_best:.2f}")

    if current_best < TARGET_MSE:
        print(f"✅ Target MSE of {TARGET_MSE} achieved! Best MSE: {current_best:.2f}")
        break
else:
    print(f"❌ Target MSE not reached after {MAX_TRIALS} trials. Best MSE: {study.best_value:.2f}")

# === Train final model with best params ===
best_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    **study.best_params
)
best_model.fit(X_train, y_train)

# === Final evaluation ===
y_pred = best_model.predict(X_test)
final_mse = mean_squared_error(y_test, y_pred)
print(f"\n📊 Final Test MSE with best model: {final_mse:.2f}")
print(f"🎯 Best Params: {study.best_params}")
