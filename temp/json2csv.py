import json
import pandas as pd

# Load nested JSON
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Flatten
records = []
for entry in data:
    row = entry['input']
    row['expected_output'] = entry['expected_output']
    records.append(row)

# Convert to DataFrame and export
df = pd.DataFrame(records)
df.to_csv('public_cases_flat.csv', index=False)
