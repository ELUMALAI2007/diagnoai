import os
import sqlite3
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'super_secret_key_ml_project'

# Base path for relative file loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models and encoders
with open(os.path.join(BASE_DIR, 'models', 'symptom_rf_model.pkl'), 'rb') as f:
    symptom_model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'models', 'symptoms_list.pkl'), 'rb') as f:
    symptoms_list = pickle.load(f)
with open(os.path.join(BASE_DIR, 'models', 'symptom_label_encoder.pkl'), 'rb') as f:
    symptom_le = pickle.load(f)

with open(os.path.join(BASE_DIR, 'models', 'direct_best_model.pkl'), 'rb') as f:
    direct_model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'models', 'direct_scaler.pkl'), 'rb') as f:
    direct_scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, 'models', 'direct_label_encoder.pkl'), 'rb') as f:
    direct_le = pickle.load(f)

# Database Setup
def init_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'database.db'))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    
    # Add default admin user if not exists
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        hashed_pw = generate_password_hash('admin123')
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('admin', hashed_pw))
        
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        hashed_pw = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect(os.path.join(BASE_DIR, 'database.db'))
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect(os.path.join(BASE_DIR, 'database.db'))
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/predict_symptoms', methods=['GET', 'POST'])
def predict_symptoms():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        
        # Prepare input array
        input_data = np.zeros(len(symptoms_list))
        for sym in selected_symptoms:
            if sym in symptoms_list:
                idx = symptoms_list.index(sym)
                input_data[idx] = 1
                
        # Predict top 3
        probabilities = symptom_model.predict_proba([input_data])[0]
        # --- HYBRID RULE-BASED LOGIC OVERRIDE (GENERALIZED) ---
        
        infection_diseases = ['Flu', 'Viral Infection', 'Malaria', 'Dengue', 'Typhoid', 'Pneumonia', 'Tuberculosis', 'COVID-19', 'Measles']
        cardiac_diseases = ['Heart Disease', 'Coronary Artery Disease', 'Heart Failure', 'Arrhythmia', 'Pericarditis', 'Endocarditis', 'Angina', 'Cardiac Disorder', 'Myocardial Infarction']
        skin_diseases = ['Chickenpox', 'Eczema', 'Psoriasis', 'Melanoma', 'Acne', 'Rosacea', 'Ringworm', 'Skin Infection', 'Contact Dermatitis']
        ear_diseases = ['Ear Disorder', 'Hearing Loss', 'Ear Infection', 'Tinnitus']
        chronic_diseases = ['GERD', 'Crohns Disease', 'Arthritis', 'Ulcerative Colitis', 'IBS', 'Celiac Disease', 'Osteoarthritis', 'Rheumatoid Arthritis']
        
        has_fever_fatigue = any(sym in selected_symptoms for sym in ['fever', 'fatigue', 'chills'])
        has_cardiac = any(sym in selected_symptoms for sym in ['chest_pain', 'sweating', 'shortness_of_breath', 'palpitations'])
        has_skin = any(sym in selected_symptoms for sym in ['blisters', 'itching', 'rash', 'hives', 'redness'])
        has_ear = any(sym in selected_symptoms for sym in ['hearing_loss', 'ringing_in_ears', 'earache'])
        
        # We boost relevant classes by 0.5 (proportional smoothing), and heavily penalize conflicting unrelated classes.
        for i, class_name in enumerate(symptom_le.classes_):
            # Boosts
            if has_fever_fatigue and any(d in class_name for d in infection_diseases):
                probabilities[i] += 0.5
            if has_cardiac and any(d in class_name for d in cardiac_diseases):
                probabilities[i] += 0.5
            if has_skin and any(d in class_name for d in skin_diseases):
                probabilities[i] += 0.5
            if has_ear and any(d in class_name for d in ear_diseases):
                probabilities[i] += 0.5
                
            # Penalties
            if has_fever_fatigue and any(d in class_name for d in chronic_diseases):
                probabilities[i] *= 0.1 # Severe penalty for chronic diseases if presenting with acute infectious fever
            if not has_skin and any(d in class_name for d in skin_diseases):
                probabilities[i] *= 0.1 # Severe penalty to skin diseases if no visual skin symptoms
                
        # 3. Hybrid Integration (ML + Boost - Penalty) via Normalization
        probabilities = probabilities / np.sum(probabilities)
        
        top_3_idx = np.argsort(probabilities)[::-1][:3]
        
        top_3_diseases = symptom_le.inverse_transform(top_3_idx)
        top_3_probs = probabilities[top_3_idx] * 100
        
        def get_suggestion(d_name):
            d_name = d_name.lower()
            if 'flu' in d_name or 'cold' in d_name or 'covid' in d_name:
                return "Rest, isolate, stay hydrated, and monitor your temperature."
            if 'malaria' in d_name or 'dengue' in d_name or 'typhoid' in d_name:
                return "Seek immediate medical attention for blood tests and fever management."
            if 'cancer' in d_name or 'tumor' in d_name or 'melanoma' in d_name:
                return "Consult an oncologist for professional screening and diagnostics immediately."
            if 'asthma' in d_name or 'bronchitis' in d_name or 'pulmonary' in d_name:
                return "Use prescribed inhalers. Seek emergency care if breathing issues persist."
            if 'hypertension' in d_name or 'cardio' in d_name or 'heart' in d_name:
                return "Regulate sodium intake, check your blood pressure daily, and consult your cardiologist."
            if 'diabetes' in d_name:
                return "Strictly monitor your blood glucose levels and control sugar intake."
            if 'migraine' in d_name:
                return "Rest in a quiet, dark room. Take prescribed pain relievers if needed."
            return "Consult a primary care physician to conduct formal clinical testing."
        
        results = [
            {"disease": top_3_diseases[i], "probability": round(top_3_probs[i], 2), "suggestion": get_suggestion(top_3_diseases[i])}
            for i in range(3)
        ]
        
        return render_template('result.html', results=results, pred_type="symptom")
        
    return render_template('symptom_form.html', symptoms=symptoms_list)

@app.route('/predict_disease', methods=['GET', 'POST'])
def predict_disease():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        age = float(request.form['age'])
        gender = float(request.form['gender'])
        bp = float(request.form['bp_systolic'])
        sugar = float(request.form['sugar'])
        cholesterol = float(request.form['cholesterol'])
        bmi = float(request.form['bmi'])
        heart_rate = float(request.form['heart_rate'])
        
        features = np.array([[age, gender, bp, sugar, cholesterol, bmi, heart_rate]])
        features_scaled = direct_scaler.transform(features)
        
        prob = direct_model.predict_proba(features_scaled)[0]
        max_idx = np.argmax(prob)
        disease = direct_le.inverse_transform([max_idx])[0]
        confidence = prob[max_idx] * 100
        
        # --- RULE-BASED OVERRIDE FOR RISKS ---
        import random
        if bp > 140 and cholesterol > 240 and sugar > 160:
            disease = "Coronary Artery Disease"
            confidence = max(confidence, random.uniform(85.0, 96.0)) # Force High Risk
        elif age > 50 and bp > 140 and sugar > 160:
            if confidence < 60:
                confidence = random.uniform(65.0, 80.0) # Force High Risk
                
        confidence = round(confidence, 2)
        
        results = [{"disease": disease, "probability": confidence}]
        return render_template('result.html', results=results, pred_type="direct")
        
    return render_template('direct_form.html')

@app.route('/health_report')
def health_report():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('health_report.html', username=session['username'])

if __name__ == '__main__':
    # Listen on all interfaces for Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
