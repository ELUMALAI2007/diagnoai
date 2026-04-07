import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def train_symptom_model():
    print("Training Symptom-Based Model...")
    df = pd.read_csv('datasets/symptom_data.csv')
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    with open('models/symptom_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    symptoms_list = X.columns.tolist()
    with open('models/symptoms_list.pkl', 'wb') as f:
        pickle.dump(symptoms_list, f)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=250, max_depth=40, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    
    print(f"Symptom Models - RF: {rf_acc:.4f}, DT: {dt_acc:.4f}, NB: {nb_acc:.4f}")
    
    best_model = rf # Choosing RF as per requirement
    
    # Metrics
    precision = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)
    print(f"RF Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Save model
    with open('models/symptom_rf_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    # Feature Importance Plot
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20] # Top 20
    plt.figure(figsize=(10,6))
    plt.title("Top 20 Feature Importances - Symptom Model")
    plt.bar(range(20), importances[indices], align="center")
    plt.xticks(range(20), [symptoms_list[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('plots/symptom_feature_importance.png')
    plt.close()

def train_direct_model():
    print("Training Direct Disease Model...")
    df = pd.read_csv('datasets/direct_disease_data.csv')
    
    X = df.drop(['Target'], axis=1)
    y = df['Target']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    with open('models/direct_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    with open('models/direct_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=250, max_depth=40, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    print(f"Direct Models - LR: {lr_acc:.4f}, RF: {rf_acc:.4f}")
    
    # Select better model
    if rf_acc > lr_acc:
        best_model = rf
        y_pred = rf_pred
        print("Selected Random Forest")
    else:
        best_model = lr
        y_pred = lr_pred
        print("Selected Logistic Regression")
        
    # Metrics
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Best Model Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Save model
    with open('models/direct_best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
if __name__ == "__main__":
    train_symptom_model()
    train_direct_model()
    print("Model training complete.")
