import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
cleveland_df = pd.read_csv("Heart_disease_cleveland_new.csv")

# Preprocess
X = cleveland_df.drop('target', axis=1)
y = cleveland_df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump((model, scaler, X.columns.tolist()), "random_forest_model.pkl")
print("✅ Model saved as random_forest_model.pkl")
