import joblib
import numpy as np
import json

# Load the scaler to get the feature names
scaler = joblib.load('model/scaler(1).pkl')
features = [str(f) for f in getattr(scaler, 'feature_names_in_', [])]

# Generate random values for each feature
values = np.random.rand(len(features)).tolist()
data = dict(zip(features, values))

# Write to a JSON file
with open('test_input.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Generated test_input.json with random values for all features.")