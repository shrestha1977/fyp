# retrain_dementia_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ---------------- Step 1: Load Dataset ----------------
# CSV must have columns: age, avg_reaction_time, correct_answers, wrong_answers, stroop_score, dementia
df = pd.read_csv("stroop_dataset.csv")

# Optional: check balance
print("Dementia class distribution:\n", df['dementia'].value_counts())

# ---------------- Step 2: Features and Target ----------------
X = df[['age', 'avg_reaction_time', 'correct_answers', 'wrong_answers', 'stroop_score']]
y = df['dementia']

# ---------------- Step 3: Feature Scaling ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for Streamlit app
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("Scaler saved as scaler.pkl")

# ---------------- Step 4: Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Step 5: Train Model ----------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------- Step 6: Evaluate ----------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------- Step 7: Save Model ----------------
pickle.dump(model, open("dementia_model.pkl", "wb"))
print("Model saved as dementia_model.pkl")
