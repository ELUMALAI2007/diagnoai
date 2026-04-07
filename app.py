import os
import sqlite3
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'super_secret_key_ml_project'

# -----------------------------
# BASE PATH
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# AUTO TRAIN MODEL
# -----------------------------
model_path = os.path.join(MODEL_DIR, 'symptom_rf_model.pkl')

if not os.path.exists(model_path):
    import train_model
    train_model.main()   # 🔥 Force training

# -----------------------------
# LOAD MODELS
# -----------------------------
with open(os.path.join(MODEL_DIR, 'symptom_rf_model.pkl'), 'rb') as f:
    symptom_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'symptoms_list.pkl'), 'rb') as f:
    symptoms_list = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'symptom_label_encoder.pkl'), 'rb') as f:
    symptom_le = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'direct_best_model.pkl'), 'rb') as f:
    direct_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'direct_scaler.pkl'), 'rb') as f:
    direct_scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'direct_label_encoder.pkl'), 'rb') as f:
    direct_le = pickle.load(f)

# -----------------------------
# DATABASE
# -----------------------------
def init_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'database.db'))
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')

    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  ('admin', generate_password_hash('admin123')))

    conn.commit()
    conn.close()

init_db()

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        conn = sqlite3.connect(os.path.join(BASE_DIR, 'database.db'))
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?",
                  (request.form['username'],))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], request.form['password']):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('dashboard'))

        flash('Invalid login', 'danger')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', username=session.get('username', 'User'))

# -----------------------------
# SYMPTOM PREDICTION
# -----------------------------
@app.route('/predict_symptoms', methods=['GET', 'POST'])
def predict_symptoms():
    if request.method == 'POST':
        selected = request.form.getlist('symptoms')

        input_data = np.zeros(len(symptoms_list))
        for s in selected:
            if s in symptoms_list:
                input_data[symptoms_list.index(s)] = 1

        probs = symptom_model.predict_proba([input_data])[0]

        # Rule boost
        if 'chest_pain' in selected and 'sweating' in selected:
            for i, d in enumerate(symptom_le.classes_):
                if "heart" in d.lower():
                    probs[i] += 0.5

        probs = probs / np.sum(probs)

        top = np.argsort(probs)[::-1][:3]

        results = [{
            "disease": symptom_le.inverse_transform([i])[0],
            "probability": round(probs[i] * 100, 2)
        } for i in top]

        return render_template('result.html', results=results)

    return render_template('symptom_form.html', symptoms=symptoms_list)

# -----------------------------
# DIRECT PREDICTION
# -----------------------------
@app.route('/predict_disease', methods=['GET', 'POST'])
def predict_disease():
    if request.method == 'POST':
        data = np.array([[ 
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['bp_systolic']),
            float(request.form['sugar']),
            float(request.form['cholesterol']),
            float(request.form['bmi']),
            float(request.form['heart_rate'])
        ]])

        data = direct_scaler.transform(data)
        probs = direct_model.predict_proba(data)[0]

        idx = np.argmax(probs)
        disease = direct_le.inverse_transform([idx])[0]
        prob = probs[idx] * 100

        if float(request.form['bp_systolic']) > 140:
            prob = max(prob, 75)

        return render_template('result.html',
                               results=[{"disease": disease,
                                         "probability": round(prob, 2)}])

    return render_template('direct_form.html')

# -----------------------------
# RUN
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
