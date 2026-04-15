#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data', 'prescriptions.csv')
MODELS = os.path.join(BASE, 'models')
os.makedirs(MODELS, exist_ok=True)

print("=" * 60)
print("  TRAINING DISEASE PREDICTION MODEL")
print("=" * 60)

# Load data
print("\n[1] Loading CSV file...")
df = pd.read_csv(DATA)
print(f"    Loaded {len(df)} records")
print(f"    Columns: {list(df.columns)}")

# Remove patient_id column if it exists
if 'patient_id' in df.columns:
    df = df.drop('patient_id', axis=1)
    print("    Removed patient_id column")

print(f"\n[2] Disease distribution in training data:")
disease_counts = df['disease'].value_counts()
for disease, count in disease_counts.items():
    print(f"    {disease}: {count} records")

# Encode categorical variables
print("\n[3] Encoding categorical variables...")
le_gender = LabelEncoder()
le_medicine = LabelEncoder()
le_disease = LabelEncoder()

df['gender_enc'] = le_gender.fit_transform(df['gender'])
df['medicine_enc'] = le_medicine.fit_transform(df['medicine'])
df['disease_enc'] = le_disease.fit_transform(df['disease'])

# Save encoders
joblib.dump(le_gender, os.path.join(MODELS, 'le_gender.pkl'))
joblib.dump(le_medicine, os.path.join(MODELS, 'le_medicine.pkl'))
joblib.dump(le_disease, os.path.join(MODELS, 'le_disease.pkl'))
print("    Saved encoders to models/")

# Prepare features
features = ['age', 'gender_enc', 'medicine_enc', 'dosage_mg', 'frequency', 'month']
X = df[features]
y = df['disease_enc']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODELS, 'scaler.pkl'))
print("    Saved scaler to models/")

# Train model
print("\n[4] Training Random Forest model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n[5] Model Accuracy: {accuracy:.1f}%")

# Save model
joblib.dump(model, os.path.join(MODELS, 'random_forest.pkl'))
print("    Saved model to models/random_forest.pkl")

# Test prediction
print("\n[6] Testing model with sample inputs:")
test_cases = [
    {"age": 8, "gender": "Male", "medicine": "Paracetamol", "dosage_mg": 250, "frequency": 3, "month": 7, "expected": "Fever"},
    {"age": 45, "gender": "Female", "medicine": "Metformin", "dosage_mg": 500, "frequency": 2, "month": 1, "expected": "Diabetes"},
    {"age": 65, "gender": "Male", "medicine": "Amlodipine", "dosage_mg": 5, "frequency": 1, "month": 3, "expected": "Hypertension"},
]

for test in test_cases:
    gender_enc = le_gender.transform([test["gender"]])[0]
    medicine_enc = le_medicine.transform([test["medicine"]])[0]
    
    features = np.array([[
        test["age"], gender_enc, medicine_enc, 
        test["dosage_mg"], test["frequency"], test["month"]
    ]])
    features_scaled = scaler.transform(features)
    
    pred_enc = model.predict(features_scaled)[0]
    pred_disease = le_disease.inverse_transform([pred_enc])[0]
    
    probs = model.predict_proba(features_scaled)[0]
    confidence = max(probs) * 100
    
    print(f"    {test['expected']} test → Predicted: {pred_disease} ({confidence:.0f}% confidence)")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print("  Run: python -m app.app")
print("=" * 60)