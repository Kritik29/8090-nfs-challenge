import json
import pandas as pd
import plotly.express as px

# === Step 1: Load and flatten the JSON data ===
with open('public_cases.json', 'r') as f:
    raw_data = json.load(f)

# Flatten into a DataFrame
records = []
for case in raw_data:
    row = case['input']
    row['reimbursement'] = case['expected_output']
    records.append(row)

df = pd.DataFrame(records)

# === Step 2: Plot for each selected trip_duration_days ===
for duration in [1, 3, 5, 7]:
    subset = df[df['trip_duration_days'] == duration]

    if subset.empty:
        print(f"No data for trip_duration_days = {duration}")
        continue

    fig = px.scatter_3d(
        subset,
        x='miles_traveled',
        y='total_receipts_amount',
        z='reimbursement',
        color='reimbursement',
        color_continuous_scale='Plasma',
        title=f'3D Plot: Trip Duration = {duration} Days',
        labels={
            'miles_traveled': 'Miles',
            'total_receipts_amount': 'Receipts ($)',
            'reimbursement': 'Reimbursement ($)'
        }
    )

    fig.update_traces(marker=dict(size=4))
    fig.show()
