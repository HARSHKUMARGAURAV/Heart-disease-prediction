# ==============================
# Heart Disease Prediction Project
# ==============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import shap

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("data/heart.csv")

print("Dataset Preview:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

# ==============================
# Data Preprocessing
# ==============================

# Handle Missing Values
print("\nMissing Values:\n", df.isnull().sum())
df = df.dropna()

# Encode Categorical Features
le = LabelEncoder()
categorical_cols = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Split Features & Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# Model Training
# ==============================

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)

# ==============================
# Model Evaluation
# ==============================

rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, xgb_pred))

# ==============================
# Feature Importance
# ==============================
importances = rf.feature_importances_
feat_importance = pd.Series(importances, index=X.columns)

feat_importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.savefig("outputs/plots/feature_importance.png")
plt.show()

# ==============================
# SHAP Explainability
# ==============================
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# ==============================
# Sample Prediction
# ==============================
sample = X_test[0].reshape(1, -1)
prediction = xgb.predict(sample)

print("\nSample Prediction:", prediction)

# ==============================
# Confusion Matrix
# ==============================
cm = confusion_matrix(y_test, xgb_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("outputs/plots/confusion_matrix.png")
plt.show()
