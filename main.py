
import os
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



os.makedirs("outputs/plots", exist_ok=True)



def load_data(path="data/heart.csv"):
    df = pd.read_csv(path)
    print("\nDataset Preview:\n", df.head())
    print("\nDataset Info:\n")
    print(df.info())
    return df



def preprocess_data(df):
    print("\nMissing Values:\n", df.isnull().sum())

df = df.dropna()


le = LabelEncoder()
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for col in categorical_cols:
 df[col] = le.fit_transform(df[col])

 # Split features and target
   X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

return X, y



def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

 xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
 xgb.fit(X_train, y_train)

 return rf, xgb

def evaluate_models(rf, xgb, X_test, y_test):
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)

 print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
 print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

print("\nClassification Report (XGBoost):\n")
print(classification_report(y_test, xgb_pred))

return rf_pred, xgb_pred


def plot_feature_importance(rf, feature_names):
    importances = rf.feature_importances_
    feat_importance = pd.Series(importances, index=feature_names)

plt.figure()
feat_importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")

plt.savefig("outputs/plots/feature_importance.png")
plt.show()


def shap_explain(xgb, X_test, feature_names):
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

 plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

plt.savefig("outputs/plots/confusion_matrix.png")
    plt.show()


# ==============================
# Main Function
# ==============================
def main():
    # Load data
    df = load_data()

 # Preprocess
X, y = preprocess_data(df)

 # Train-test split
 X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
    )

# Scaling
 scaler = StandardScaler()
 X_train = scaler.fit_transform(X_train)
 X_test = scaler.transform(X_test)

    # Train models
rf, xgb = train_models(X_train, y_train)

    # Evaluate
 rf_pred, xgb_pred = evaluate_models(rf, xgb, X_test, y_test)

    # Feature importance
 plot_feature_importance(rf, X.columns)

    # SHAP
 shap_explain(xgb, X_test, X.columns)

    # Sample prediction
sample = X_test[0].reshape(1, -1)
    prediction = xgb.predict(sample)
    print("\nSample Prediction:", prediction)

    # Confusion matrix
 plot_confusion_matrix(y_test, xgb_pred)


if __name__ == "__main__":
    main()
