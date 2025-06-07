import json
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
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

# === Plot with Plotly ===
fig = px.scatter_3d(
    df,
    x='trip_duration_days',
    y='miles_traveled',
    z='total_receipts_amount',
    color='reimbursement',
    color_continuous_scale='Viridis',
    title='Interactive 3D Scatter: Inputs Colored by Reimbursement',
    labels={
        'trip_duration_days': 'Duration (days)',
        'miles_traveled': 'Miles',
        'total_receipts_amount': 'Receipts ($)',
        'reimbursement': 'Reimbursement ($)'
    }
)

fig.update_traces(marker=dict(size=4))
fig.show()

# Features and target
X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
y = df['reimbursement']

# Train/test split (optional but recommended)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Coefficients for the linear function:
print('Function:')
print(f'reimbursement = {model.intercept_:.2f} + '
      f'{model.coef_[0]:.2f} * trip_duration_days + '
      f'{model.coef_[1]:.2f} * miles_traveled + '
      f'{model.coef_[2]:.2f} * total_receipts_amount')