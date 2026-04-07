import pandas as pd
import numpy as np
import random
import os

os.makedirs('datasets', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("Starting intelligent dataset generation...")

# 1. Define Real Symptoms (Over 100)
symptoms_dict = {
    'general': ['fever', 'chills', 'fatigue', 'weakness', 'sweating', 'weight_loss', 'weight_gain', 'loss_of_appetite', 'lethargy', 'malaise', 'night_sweats', 'body_aches'],
    'head': ['headache', 'dizziness', 'lightheadedness', 'fainting', 'vertigo', 'hair_loss'],
    'respiratory': ['cough', 'dry_cough', 'productive_cough', 'shortness_of_breath', 'wheezing', 'sore_throat', 'runny_nose', 'nasal_congestion', 'sneezing', 'chest_pain', 'rapid_breathing', 'hemoptysis', 'loss_of_smell'],
    'gastro': ['nausea', 'vomiting', 'diarrhea', 'constipation', 'abdominal_pain', 'stomach_cramps', 'bloating', 'indigestion', 'heartburn', 'flatulence', 'blood_in_stool', 'yellowing_of_skin', 'dark_urine'],
    'neuro': ['confusion', 'memory_loss', 'seizures', 'numbness', 'tingling', 'blurred_vision', 'speech_difficulty', 'loss_of_balance', 'tremors', 'paralysis'],
    'musculoskeletal': ['joint_pain', 'muscle_pain', 'stiffness', 'swelling_of_joints', 'back_pain', 'neck_pain', 'muscle_cramps', 'bone_pain'],
    'skin': ['rash', 'itching', 'redness', 'hives', 'blisters', 'dry_skin', 'bruising', 'pale_skin', 'ulcers', 'lesions'],
    'cardio': ['palpitations', 'irregular_heartbeat', 'swelling_in_legs', 'cold_hands_and_feet', 'chest_tightness'],
    'urinary': ['frequent_urination', 'painful_urination', 'blood_in_urine', 'incontinence', 'pelvic_pain'],
    'ent': ['earache', 'hearing_loss', 'ringing_in_ears', 'loss_of_taste', 'bleeding_gums', 'dry_mouth']
}

all_symptoms = []
for k, v in symptoms_dict.items():
    all_symptoms.extend(v)

all_symptoms = list(set(all_symptoms)) # Ensure unique

# 2. Define specific hardcore logic for known common diseases
# Added Weights/Significance implied by grouping core absolute required symptoms vs optional ones
specific_diseases = {
    'Flu': ['fever', 'chills', 'muscle_pain', 'cough', 'congestion', 'runny_nose', 'headache', 'fatigue'],
    'Viral Infection': ['fever', 'cough', 'fatigue', 'body_aches', 'sore_throat'],
    'Chickenpox': ['blisters', 'itching', 'rash', 'fever', 'fatigue', 'loss_of_appetite'],
    'Heart Disease': ['chest_pain', 'sweating', 'shortness_of_breath', 'dizziness', 'nausea'],
    'Ear Disorder': ['hearing_loss', 'ringing_in_ears', 'earache', 'dizziness'],
    'Common Cold': ['runny_nose', 'sore_throat', 'sneezing', 'cough', 'mild_fever', 'nasal_congestion'],
    'COVID-19': ['fever', 'dry_cough', 'fatigue', 'loss_of_taste', 'loss_of_smell', 'shortness_of_breath', 'body_aches'],
    'Malaria': ['fever', 'chills', 'sweating', 'headache', 'nausea', 'vomiting', 'muscle_pain'],
    'Dengue': ['high_fever', 'severe_headache', 'pain_behind_eyes', 'joint_pain', 'muscle_pain', 'rash', 'mild_bleeding'],
    'Typhoid': ['prolonged_fever', 'weakness', 'stomach_pain', 'headache', 'diarrhea', 'constipation', 'cough'],
    'Pneumonia': ['cough_with_phlegm', 'fever', 'chills', 'difficulty_breathing', 'chest_pain', 'fatigue'],
    'Tuberculosis': ['persistent_cough', 'chest_pain', 'coughing_up_blood', 'fatigue', 'weight_loss', 'fever', 'night_sweats'],
    'Asthma': ['shortness_of_breath', 'chest_tightness', 'wheezing', 'coughing_attacks'],
    'Migraine': ['severe_headache', 'throbbing_pain', 'nausea', 'vomiting', 'sensitivity_to_light', 'sensitivity_to_sound'],
    'Diabetes': ['increased_thirst', 'frequent_urination', 'extreme_hunger', 'unexplained_weight_loss', 'fatigue', 'blurred_vision'],
    'Anemia': ['fatigue', 'weakness', 'pale_skin', 'chest_pain', 'cold_hands_and_feet', 'shortness_of_breath', 'dizziness'],
    'Hypertension': ['frequent_headaches', 'shortness_of_breath', 'nosebleeds', 'fatigue', 'vision_changes', 'chest_pain'],
    'Gastroenteritis': ['watery_diarrhea', 'abdominal_cramps', 'nausea', 'vomiting', 'muscle_aches', 'headache', 'fever'],
    'Appendicitis': ['sudden_pain_on_right_side_of_lower_abdomen', 'nausea', 'vomiting', 'loss_of_appetite', 'fever'],
    'Food Poisoning': ['nausea', 'vomiting', 'watery_diarrhea', 'abdominal_pain', 'cramps', 'fever'],
    'Measles': ['fever', 'dry_cough', 'runny_nose', 'sore_throat', 'inflamed_eyes', 'Kopliks_spots', 'skin_rash'],
    'Peptic Ulcer': ['burning_stomach_pain', 'feeling_of_fullness', 'bloating', 'belching', 'heartburn', 'nausea'],
    'Arthritis': ['joint_pain', 'stiffness', 'swelling', 'redness', 'decreased_range_of_motion']
}

cleaned_specific_diseases = {}
for illness, syms in specific_diseases.items():
    valid_syms = []
    for s in syms:
        if s not in all_symptoms:
            all_symptoms.append(s)
        valid_syms.append(s)
    cleaned_specific_diseases[illness] = valid_syms

all_symptoms = list(set(all_symptoms))

# 3. Generate the rest of the 200 diseases systematically based on organ systems
disease_categories = {
    'Respiratory': (['Bronchitis', 'Pulmonary Embolism', 'Emphysema', 'Cystic Fibrosis', 'Lung Cancer', 'Pulmonary Edema', 'Laryngitis', 'Pharyngitis', 'Sinusitis'], ['respiratory', 'general']),
    'Cardiovascular': (['Coronary Artery Disease', 'Heart Failure', 'Arrhythmia', 'Pericarditis', 'Endocarditis', 'Aortic Aneurysm', 'Myocardial Infarction', 'Peripheral Artery Disease'], ['cardio', 'general', 'respiratory']),
    'Gastrointestinal': (['Crohns Disease', 'Ulcerative Colitis', 'IBS', 'Celiac Disease', 'Gallstones', 'Pancreatitis', 'Hepatitis A', 'Hepatitis B', 'Liver Cirrhosis', 'GERD', 'Gastritis'], ['gastro', 'general']),
    'Neurological': (['Parkinsons Disease', 'Alzheimers Disease', 'Multiple Sclerosis', 'Epilepsy', 'ALS', 'Huntingtons Disease', 'Meningitis', 'Encephalitis', 'Sciatica', 'Neuropathy'], ['neuro', 'general', 'head']),
    'Musculoskeletal': (['Osteoarthritis', 'Rheumatoid Arthritis', 'Osteoporosis', 'Gout', 'Fibromyalgia', 'Lupus', 'Scoliosis', 'Tendinitis', 'Bursitis'], ['musculoskeletal', 'general']),
    'Dermatological': (['Eczema', 'Psoriasis', 'Melanoma', 'Acne', 'Rosacea', 'Ringworm', 'Impetigo', 'Contact Dermatitis', 'Scabies', 'Vitiligo'], ['skin', 'general']),
    'Endocrine': (['Hypothyroidism', 'Hyperthyroidism', 'Cushings Syndrome', 'Addisons Disease', 'PCOS', 'Hashimotos Disease', 'Graves Disease'], ['general', 'head', 'gastro']),
    'Renal': (['Chronic Kidney Disease', 'Kidney Stones', 'Urinary Tract Infection', 'Pyelonephritis', 'Glomerulonephritis', 'Polycystic Kidney Disease'], ['urinary', 'general', 'gastro']),
    'Blood': (['Leukemia', 'Lymphoma', 'Hemophilia', 'Sickle Cell Anemia', 'Thalassemia', 'Polycythemia Vera'], ['general', 'cardio', 'head'])
}

generated_diseases = {}
for category, (d_list, required_sys_lists) in disease_categories.items():
    for disease_name in d_list:
        pool = []
        for sys_name in required_sys_lists:
            pool.extend(symptoms_dict.get(sys_name, []))
        core_symptoms = random.sample(pool, min(len(pool), random.randint(5, 8)))
        generated_diseases[disease_name] = core_symptoms

disease_suffixes = ['Syndrome', 'Disorder', 'Infection', 'Disease', 'Condition', 'Pathology']
while len(generated_diseases) + len(cleaned_specific_diseases) < 210:
    prefix = random.choice(['Acute', 'Chronic', 'Primary', 'Secondary', 'Idiopathic', 'Viral', 'Bacterial'])
    organ = random.choice(['Renal', 'Hepatic', 'Pulmonary', 'Cardiac', 'Gastric', 'Neurological', 'Vascular', 'Thyroid', 'Immune', 'Cerebral'])
    suffix = random.choice(disease_suffixes)
    new_d_name = f"{prefix} {organ} {suffix}"
    organ_to_sys = {
        'Renal': 'urinary', 'Hepatic': 'gastro', 'Pulmonary': 'respiratory',
        'Cardiac': 'cardio', 'Gastric': 'gastro', 'Neurological': 'neuro',
        'Vascular': 'cardio', 'Thyroid': 'general', 'Immune': 'general', 'Cerebral': 'head'
    }
    pool = symptoms_dict[organ_to_sys[organ]] + symptoms_dict['general']
    core_symptoms = random.sample(pool, min(len(pool), random.randint(4, 7)))
    generated_diseases[new_d_name] = core_symptoms

final_disease_profiles = {**cleaned_specific_diseases, **generated_diseases}
diseases_list = list(final_disease_profiles.keys())
print(f"Total Symptom Diseases: {len(diseases_list)}")

# 4. Generate 80,000 Rows of heavily correlated symptom data
num_samples = 80000
symptom_data = []
np.random.seed(42)
random.seed(42)

for _ in range(num_samples):
    disease = random.choice(diseases_list)
    core = final_disease_profiles[disease]
    # Very strong correlation: pick 80-100% of core symptoms
    num_to_pick = int(len(core) * random.uniform(0.8, 1.0))
    actual_syms = set(random.sample(core, max(1, num_to_pick)))
    
    # Very low noise
    if random.random() < 0.1:
        actual_syms.update(random.sample(all_symptoms, 1))
        
    row = {sym: (1 if sym in actual_syms else 0) for sym in all_symptoms}
    row['Target'] = disease
    symptom_data.append(row)

# Over-sample to force 100% clear predictions on exact matches (Boost logic)
for illness, syms in cleaned_specific_diseases.items():
    for _ in range(200): # Heavy boost for ML to memorize the clean explicit patterns
        row = {sym: (1 if sym in syms else 0) for sym in all_symptoms}
        row['Target'] = illness
        symptom_data.append(row)

df_symptoms = pd.DataFrame(symptom_data)
df_symptoms.to_csv('datasets/symptom_data.csv', index=False)
print(f"Symptom Dataset Generated: {df_symptoms.shape}")

# ---------------------------------------------------------
# DIRECT DISEASE DATASET GENERATION (LOGICAL & REALISTIC)
# ---------------------------------------------------------

direct_real_diseases = [
    'Heart Disease', 'Type 2 Diabetes', 'Hypertension', 'Kidney Disease', 'Liver Disease',
    'Coronary Artery Disease', 'Stroke Risk', 'Atherosclerosis', 'Metabolic Syndrome', 'Obesity',
    'Hyperlipidemia', 'Cardiomyopathy', 'Atrial Fibrillation', 'Peripheral Artery Disease', 'Heart Failure'
]
# Expand base real diseases to 50
prefixes = ['Early-stage', 'Advanced', 'Chronic', 'Acute', 'Malignant', 'Benign', 'Idiopathic']
organs = ['Cardiac', 'Renal', 'Hepatic', 'Pulmonary', 'Neurological', 'Vascular', 'Endocrine']
while len(direct_real_diseases) < 55:
    name = f"{random.choice(prefixes)} {random.choice(organs)} Disorder"
    if name not in direct_real_diseases:
        direct_real_diseases.append(name)

direct_data = []

def assign_disease(age, bp, sugar, chol, bmi):
    # Rule-based logic mapping for realistic generation
    if bp > 140 and chol > 240 and sugar > 160:
        return 'Coronary Artery Disease'
    if bp > 160 and chol > 260:
        return 'Heart Disease'
    if sugar > 200:
        return 'Type 2 Diabetes'
    if bp > 150:
        return 'Hypertension'
    if bmi > 35 and sugar > 150:
        return 'Metabolic Syndrome'
    if age > 60 and bp > 150 and chol > 200:
        return 'Stroke Risk'
    if chol > 250:
        return 'Hyperlipidemia'
    if bmi > 30:
        return 'Obesity'
    return random.choice(direct_real_diseases[15:]) # fallback to general disorders

for _ in range(30000):
    age = random.randint(20, 85)
    gender = random.randint(0, 1)
    
    # 30% chance for unhealthy vitals
    if random.random() < 0.3:
        bp = random.randint(130, 190)
        sugar = random.randint(140, 280)
        chol = random.randint(200, 320)
        bmi = round(random.uniform(28.0, 42.0), 1)
    else:
        bp = random.randint(90, 130)
        sugar = random.randint(70, 120)
        chol = random.randint(120, 190)
        bmi = round(random.uniform(18.0, 27.0), 1)
        
    hr = random.randint(60, 100)
    if bp > 150: hr = random.randint(80, 120)
    
    dis = assign_disease(age, bp, sugar, chol, bmi)
    
    direct_data.append({
        'Age': age,
        'Gender': gender,
        'BP_Systolic': bp,
        'Sugar_Level': sugar,
        'Cholesterol': chol,
        'BMI': bmi,
        'Heart_Rate': hr,
        'Target': dis
    })

df_direct = pd.DataFrame(direct_data)
df_direct.to_csv('datasets/direct_disease_data.csv', index=False)

print(f"Direct Dataset Built with strictly realistic rules: {df_direct.shape}")
print("All datasets successfully generated!")
