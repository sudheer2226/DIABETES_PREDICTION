# ================================
# Diabetes Prediction Project
# Beginner-Friendly & Professional
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("diabetes_prediction_dataset.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)
print(data.head())

# -------------------------------
# 2. Encode Categorical Columns
# -------------------------------

# Encode gender
gender_encoder = LabelEncoder()
data["gender"] = gender_encoder.fit_transform(data["gender"])

# Encode smoking_history
smoking_encoder = LabelEncoder()
data["smoking_history"] = smoking_encoder.fit_transform(data["smoking_history"])

# -------------------------------
# 3. Split Features & Target
# -------------------------------

X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# -------------------------------
# 4. Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. Feature Scaling
# -------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Train Model (High Accuracy)
# -------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -------------------------------
# 7. Model Evaluation
# -------------------------------

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ MODEL ACCURACY:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 8. User Input Prediction
# -------------------------------

print("\n--- ENTER PATIENT DETAILS ---")

gender_input = input("Gender (Male/Female): ")
age = float(input("Age: "))
hypertension = int(input("Hypertension (0 = No, 1 = Yes): "))
heart_disease = int(input("Heart Disease (0 = No, 1 = Yes): "))
smoking_input = input("Smoking History (never / current / former / No Info): ")
bmi = float(input("BMI: "))
hba1c = float(input("HbA1c Level: "))
glucose = float(input("Blood Glucose Level: "))

# Encode inputs
gender_encoded = gender_encoder.transform([gender_input])[0]
smoking_encoded = smoking_encoder.transform([smoking_input])[0]

# Create DataFrame with correct feature names (NO WARNINGS)
input_data = pd.DataFrame(
    [[
        gender_encoded,
        age,
        hypertension,
        heart_disease,
        smoking_encoded,
        bmi,
        hba1c,
        glucose
    ]],
    columns=X.columns
)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]

# -------------------------------
# 9. Final Result
# -------------------------------

print("\n✅ FINAL RESULT:")

if prediction == 1:
    print("PERSON IS DIABETIC")
else:
    print("PERSON IS NOT DIABETIC")
