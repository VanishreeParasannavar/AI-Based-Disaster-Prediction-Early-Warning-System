import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Dummy training data: [temperature, rainfall, wind_speed]
X = np.array([
    [35, 100, 20],
    [40, 200, 50],
    [30, 50, 10],
    [45, 300, 70]
])
y = [1, 1, 0, 1]  # 1 = Disaster likely, 0 = Safe

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model to a file
with open('disaster_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as disaster_model.pkl")
