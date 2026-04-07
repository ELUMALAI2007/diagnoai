import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

os.makedirs('models', exist_ok=True)

def train_symptom_model():
    df = pd.read_csv('datasets/symptom_data.csv')

    X = df.drop('Target', axis=1)
    y = df['Target']

    le = LabelEncoder()
    y = le.fit_transform(y)

    with open('models/symptom_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    with open('models/symptoms_list.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    with open('models/symptom_rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def train_direct_model():
    df = pd.read_csv('datasets/direct_disease_data.csv')

    X = df.drop('Target', axis=1)
    y = df['Target']

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    with open('models/direct_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    with open('models/direct_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    with open('models/direct_best_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# 🔥 MAIN FUNCTION (VERY IMPORTANT)
def main():
    train_symptom_model()
    train_direct_model()
    print("Models trained successfully")

if __name__ == "__main__":
    main()
