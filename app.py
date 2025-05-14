from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import os
import joblib
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Ensure plots folder exists
if not os.path.exists('static/plots'):
    os.makedirs('static/plots')

# Load models and scaler
scaler = joblib.load('D:/FYP/saved_models/scaler.pkl')
feature_names = joblib.load('D:/FYP/saved_models/feature_names.pkl')

model_paths = {
    'Device Risk Classification': 'D:/FYP/saved_models/best_model_device_risk_classification.pkl',
    'Causality Assessment': 'D:/FYP/saved_models/best_model_causality_assessment.pkl',
    'Serious Event': 'D:/FYP/saved_models/best_model_serious_event.pkl',
    'Prolongation of Event': 'D:/FYP/saved_models/best_model_prolongation_of_event.pkl',
    'Potential Diseases or Side Effects': 'D:/FYP/saved_models/best_model_potential_diseases_or_side_effects.pkl',
    'Prevention Techniques': 'D:/FYP/saved_models/best_model_prevention_techniques.pkl'
}

models = {}
for target_name, path in model_paths.items():
    try:
        models[target_name] = joblib.load(path)
        logger.debug(f"Loaded model for {target_name}")
    except Exception as e:
        logger.error(f"Error loading {target_name}: {str(e)}")
        flash(f"Model load error for {target_name}", 'danger')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                data = pd.read_excel(file)
                generate_plots(data)
                flash('File uploaded and processed successfully!', 'success')
                return redirect(url_for('display_results'))
            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                flash(f"Processing error: {str(e)}", 'danger')
        else:
            flash('No file uploaded.', 'danger')
    return render_template('upload.html')

def generate_plots(data):
    # Strip column names
    data.rename(columns=lambda x: x.strip(), inplace=True)

    # Clear existing plots
    for f in os.listdir('static/plots'):
        path = os.path.join('static/plots', f)
        if os.path.isfile(path):
            os.remove(path)

    # Age Range
    age_bins = [0, 18, 30, 45, 60, 75, 90, 105]
    age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76-90', '91+']
    data['Age Range'] = pd.cut(data['Age of the patient'], bins=age_bins, labels=age_labels)
    age_dist = data['Age Range'].value_counts().sort_index()
    plt.figure(figsize=(10, 10))
    plt.pie(age_dist, labels=[f'{i} ({c})' for i, c in zip(age_dist.index, age_dist)], autopct='%1.1f%%',
            colors=sns.color_palette('viridis', len(age_dist)), startangle=140)
    plt.title('Age Range Distribution', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('static/plots/age_range_distribution_pie.png')
    plt.close()

    # Gender
    gender_dist = data['Gender'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(gender_dist, labels=gender_dist.index, autopct='%1.1f%%',
            colors=sns.color_palette('viridis', len(gender_dist)))
    plt.title('Gender Distribution')
    plt.tight_layout()
    plt.savefig('static/plots/gender_distribution_pie.png')
    plt.close()

    # Device MDAE
    device_dist = data['Name of the device'].value_counts().head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=device_dist.values, y=device_dist.index, color=sns.color_palette('viridis')[0])
    plt.title('Top Devices Causing MDAEs')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig('static/plots/device_distribution.png')
    plt.close()

    # Risk Classification
    risk_dist = data['Device risk classification as per India MDR 2017'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=risk_dist.index, y=risk_dist.values, color=sns.color_palette('viridis')[0])
    plt.title('Device Risk Classification')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/plots/risk_classification_distribution.png')
    plt.close()

    # Causality
    if 'Causality assessment' in data.columns:
        causality_dist = data['Causality assessment'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=causality_dist.index, y=causality_dist.values, color=sns.color_palette('viridis')[0])
        plt.title('Causality Assessment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/plots/causality_assessment_distribution.png')
        plt.close()
    else:
        flash("Missing column: 'Causality assessment'", 'warning')

    # Location
    loc_dist = data['Location of event'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=loc_dist.values, y=loc_dist.index, color=sns.color_palette('viridis')[0])
    plt.title('Location of Event')
    plt.tight_layout()
    plt.savefig('static/plots/location_distribution.png')
    plt.close()

    # Severity (Fixed)
    if 'severity' in data.columns:
        data['severity'] = data['severity'].astype(str).str.strip().str.replace(")", "", regex=False)
        sev_dist = data['severity'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(sev_dist, labels=sev_dist.index, autopct='%1.1f%%',
                colors=sns.color_palette('viridis', len(sev_dist)))
        plt.title('Severity Distribution')
        plt.tight_layout()
        plt.savefig('static/plots/severity_distribution.png')
        plt.close()
    else:
        flash("Missing column: 'severity'", 'warning')

    # Patient Outcomes
    if 'Patient Outcomes' in data.columns:
        outcome_dist = data['Patient Outcomes'].value_counts()
        plt.figure(figsize=(10, 10))
        plt.pie(outcome_dist, labels=outcome_dist.index, autopct='%1.1f%%',
                colors=sns.color_palette('viridis', len(outcome_dist)))
        plt.title('Patient Outcomes Distribution')
        plt.tight_layout()
        plt.savefig('static/plots/patient_outcomes_distribution.png')
        plt.close()
    else:
        flash("Missing column: 'Patient Outcomes'", 'warning')

    # Device vs Manufacturer
    if 'Manufacturer name' in data.columns:
        grp = data.groupby(['Manufacturer name', 'Name of the device']).size().reset_index(name='MDAE Count')
        pivot_df = grp.pivot(index='Manufacturer name', columns='Name of the device', values='MDAE Count').fillna(0)
        pivot_df['Total'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values('Total', ascending=False).drop('Total', axis=1)
        pivot_df.plot(kind='barh', stacked=True, figsize=(12, 8), colormap='viridis')
        plt.title('Top Devices with Most MDAEs by Manufacturer', fontsize=16, weight='bold')
        plt.xlabel('MDAE Count')
        plt.tight_layout()
        plt.savefig('static/plots/top_devices_per_manufacturer.png')
        plt.close()
    else:
        flash("Missing column: 'Manufacturer name'", 'warning')

@app.route('/results')
def display_results():
    return render_template('results.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predictions = {}
    if request.method == 'POST':
        try:
            device_name = request.form.get('device_name')
            location = request.form.get('location_of_event')
            age = request.form.get('age')
            gender = request.form.get('gender')
            history = request.form.get('past_history')
            event_type = request.form.get('nature_of_event')

            if None in [device_name, location, age, gender, history, event_type]:
                flash('All fields are required.', 'danger')
                return redirect(url_for('predict'))

            df_input = pd.DataFrame([{
                'Name of the device': device_name,
                'Location of event': location,
                'Age of the patient': age,
                'Gender': gender,
                'Past history': history,
                'Nature of Event': event_type
            }])

            input_encoded = pd.get_dummies(df_input)
            missing_cols = set(feature_names) - set(input_encoded.columns)
            for col in missing_cols:
                input_encoded[col] = 0
            input_encoded = input_encoded[feature_names]

            scaled = scaler.transform(input_encoded)

            for key, model in models.items():
                try:
                    pred = model.predict(scaled)[0]
                    predictions[key] = pred
                except Exception as e:
                    logger.error(f"Prediction error in {key}: {str(e)}")
                    predictions[key] = "Error"
        except Exception as e:
            logger.error(f"Prediction route error: {str(e)}")
            flash(f"Prediction error: {str(e)}", 'danger')

    return render_template('predict.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)