import json
import pandas as pd
import plotly.graph_objects as go

# Load and flatten the data
with open('public_cases.json', 'r') as f:
    raw_data = json.load(f)

# Flatten into a DataFrame
records = []
for case in raw_data:
    row = case['input']
    row['reimbursement'] = case['expected_output']
    records.append(row)

df = pd.DataFrame(records)

# Define input features to plot against reimbursement
input_vars = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']

# Store plots in a dictionary
plots = {}

for var in input_vars:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[var],
        y=df['reimbursement'],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name=f'{var} vs reimbursement'
    ))
    fig.update_layout(
        title=f'Reimbursement vs {var.replace("_", " ").title()}',
        xaxis_title=var.replace("_", " ").title(),
        yaxis_title='Reimbursement ($)',
        template='plotly_white'
    )
    plots[var] = fig
    fig.show()
