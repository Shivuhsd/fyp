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
app.secret_key = 'supersecretkey'  # Necessary for flashing messages

# Ensure the static/plots directory exists
if not os.path.exists('static/plots'):
    os.makedirs('static/plots')

# Load the pre-trained scaler, feature names, and models
scaler_filename = 'models/scaler.pkl'
feature_names_filename = 'models/feature_names.pkl'
scaler = joblib.load(scaler_filename)
feature_names = joblib.load(feature_names_filename)

model_paths = {
    'Device Risk Classification': 'models/best_model_device_risk_classification.pkl',
    'Causality Assessment': 'models/best_model_causality_assessment.pkl',
    'Serious Event': 'models/best_model_serious_event.pkl',
    'Prolongation of Event': 'models/best_model_prolongation_of_event.pkl',
    'Potential Diseases or Side Effects': 'models/best_model_potential_diseases_or_side_effects.pkl',
    'Prevention Techniques': 'models/best_model_prevention_techniques.pkl'
}

models = {}
for target_name, path in model_paths.items():
    try:
        models[target_name] = joblib.load(path)
        logger.debug(f"Loaded model for {target_name} from {path}")
    except Exception as e:
        logger.error(f"Error loading model for {target_name}: {str(e)}")
        flash(f"Error loading model for {target_name}.", 'danger')

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
                logger.error(f"Error processing file: {str(e)}")
                flash(f"Error processing file: {str(e)}", 'danger')
                return redirect(url_for('upload_file'))
        else:
            flash('No file uploaded!', 'danger')
    return render_template('upload.html')

def generate_plots(data):
    # Clear existing plots
    for filename in os.listdir('static/plots'):
        file_path = os.path.join('static/plots', filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

    # Age Range Distribution
    age_bins = [0, 18, 30, 45, 60, 75, 90, 105]
    age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76-90', '91+']
    data['Age Range'] = pd.cut(data['Age of the patient'], bins=age_bins, labels=age_labels)
    age_distribution = data['Age Range'].value_counts().sort_index()

    plt.figure(figsize=(10, 10))
    labels = [f'{label} ({count})' for label, count in zip(age_distribution.index, age_distribution)]
    plt.pie(age_distribution, labels=labels, autopct='%1.1f%%', colors=sns.color_palette('viridis', n_colors=len(age_distribution)), startangle=140)
    plt.title('Age Range Distribution of Patients', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('static/plots/age_range_distribution_pie.png')
    plt.close()

    # Gender Distribution
    gender_distribution = data['Gender'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(gender_distribution, labels=gender_distribution.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', n_colors=len(gender_distribution)))
    plt.title('Gender Distribution')
    plt.tight_layout()
    plt.savefig('static/plots/gender_distribution_pie.png')
    plt.close()

    # Most Devices that Caused MDAEs
    device_distribution = data['Name of the device'].value_counts().head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=device_distribution.values, y=device_distribution.index, color=sns.color_palette('viridis', n_colors=len(device_distribution))[0])
    plt.title('Most Devices that Caused MDAEs')
    plt.xlabel('Count')
    plt.ylabel('Device Name')
    plt.tight_layout()
    plt.savefig('static/plots/device_distribution.png')
    plt.close()

    # Device Risk Classification
    risk_classification_distribution = data['Device risk classification as per India MDR 2017'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=risk_classification_distribution.index, y=risk_classification_distribution.values, color=sns.color_palette('viridis', n_colors=len(risk_classification_distribution))[0])
    plt.title('Device Risk Classification Distribution')
    plt.xlabel('Risk Classification')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/plots/risk_classification_distribution.png')
    plt.close()

    # Causality Assessment Distribution
    if 'Causality assessment' in data.columns:
        causality_distribution = data['Causality assessment'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=causality_distribution.index, y=causality_distribution.values, color=sns.color_palette('viridis', n_colors=len(causality_distribution))[0])
        plt.title('Causality Assessment Distribution')
        plt.xlabel('Causality Assessment')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/plots/causality_assessment_distribution.png')
        plt.close()
    else:
        flash("Column 'Causality assessment' not found in the dataset.", 'warning')

    # Location of Event Distribution
    location_distribution = data['Location of event'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=location_distribution.values, y=location_distribution.index, color=sns.color_palette('viridis', n_colors=len(location_distribution))[0])
    plt.title('Location of Event Distribution')
    plt.xlabel('Count')
    plt.ylabel('Location')
    plt.tight_layout()
    plt.savefig('static/plots/location_distribution.png')
    plt.close()

    # Serious Event Distribution
    if 'Serious Event' in data.columns:
        serious_event_distribution = data['Serious Event'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(serious_event_distribution, labels=serious_event_distribution.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', n_colors=len(serious_event_distribution)))
        plt.title('Serious Event Distribution')
        plt.tight_layout()
        plt.savefig('static/plots/serious_event_distribution.png')
        plt.close()
    else:
        flash("Column 'Serious Event' not found in the dataset.", 'warning')

   # Patient Outcomes Distribution
    if 'Patient Outcomes' in data.columns:
        patient_outcomes_distribution = data['Patient Outcomes'].value_counts()
        plt.figure(figsize=(10, 10))
        plt.pie(patient_outcomes_distribution, labels=patient_outcomes_distribution.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', n_colors=len(patient_outcomes_distribution)))
        plt.title('Patient Outcomes Distribution')
        plt.tight_layout()
        plt.savefig('static/plots/patient_outcomes_distribution.png')
        plt.close()
    else:
        flash("Column 'Patient Outcomes' not found in the dataset.", 'warning')

    # Analysis: Device with Most MDAEs by Manufacturer
    # if 'Manufacturer name' in data.columns:
    #     device_manufacturer_distribution = data.groupby(['Manufacturer name', 'Name of the device']).size().reset_index(name='MDAE Count')
    #     top_devices_per_manufacturer = device_manufacturer_distribution.loc[device_manufacturer_distribution.groupby('Manufacturer name')['MDAE Count'].idxmax()]
    #     top_devices_per_manufacturer['Label'] = top_devices_per_manufacturer['Name of the device'] + ' (' + top_devices_per_manufacturer['Manufacturer name'] + ')'
    #     plt.figure(figsize=(12, 8))
    #     sns.barplot(x='MDAE Count', y='Label', data=top_devices_per_manufacturer, color=sns.color_palette('viridis', n_colors=len(top_devices_per_manufacturer))[0])
    #     plt.title('Top Devices with Most MDAEs by Manufacturer')
    #     plt.xlabel('MDAE Count')
    #     plt.ylabel('Device Name (Manufacturer)')
    #     plt.tight_layout()
    #     plt.savefig('static/plots/top_devices_per_manufacturer.png')
    #     plt.close()
    # else:
    #     flash("Column 'Manufacturer name' not found in the dataset.", 'warning')
    # Analysis: Device with Most MDAEs by Manufacturer
    if 'Manufacturer name' in data.columns:
        device_manufacturer_distribution = data.groupby(['Manufacturer name', 'Name of the device']).size().reset_index(name='MDAE Count')
        
        # Pivot the data to get manufacturers as rows and devices as columns
        pivot_df = device_manufacturer_distribution.pivot(index='Manufacturer name', columns='Name of the device', values='MDAE Count').fillna(0)
        
        # Sort manufacturers by total MDAEs
        pivot_df['Total MDAEs'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values(by='Total MDAEs', ascending=False).drop(columns='Total MDAEs')

        # Plot a horizontal stacked bar chart
        pivot_df.plot(kind='barh', stacked=True, figsize=(12, 8), colormap='viridis')
        
        plt.title('Top Devices with Most MDAEs by Manufacturer', fontsize=16, weight='bold')
        plt.xlabel('MDAE Count')
        plt.ylabel('Manufacturer')
        plt.legend(title='Device Name', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('static/plots/top_devices_per_manufacturer.png')
        plt.close()
    else:
        flash("Column 'Manufacturer name' not found in the dataset.", 'warning')



@app.route('/results')
def display_results():
    return render_template('results.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predictions = {}
    if request.method == 'POST':
        try:
            # Retrieve form data
            device_name = request.form.get('device_name')
            location_of_event = request.form.get('location_of_event')
            age = request.form.get('age')
            gender = request.form.get('gender')
            past_history = request.form.get('past_history')
            nature_of_event = request.form.get('nature_of_event')

            # Validate that none of the inputs are None (i.e., no missing data)
            if None in [device_name, location_of_event, age, gender, past_history, nature_of_event]:
                flash('Please fill out all fields.', 'danger')
                return redirect(url_for('predict'))

            # Prepare input data for the model
            input_data = pd.DataFrame({
                'Name of the device': [device_name],
                'Location of event': [location_of_event],
                'Age of the patient': [age],
                'Gender': [gender],
                'Past history': [past_history],
                'Nature of Event': [nature_of_event]
            })

            # One-hot encode the sample input data
            input_data_encoded = pd.get_dummies(input_data)

            # Create a DataFrame with the missing columns as zeros
            missing_cols = set(feature_names) - set(input_data_encoded.columns)
            missing_data = pd.DataFrame(0, index=input_data_encoded.index, columns=list(missing_cols))
            
            # Concatenate the DataFrames to include the missing columns
            input_data_encoded = pd.concat([input_data_encoded, missing_data], axis=1)

            # Reorder columns to match the feature names
            input_data_encoded = input_data_encoded[feature_names]

            # Scale the sample input data
            input_data_scaled = scaler.transform(input_data_encoded)

            # Make predictions using the pre-trained models
            for target_name, model in models.items():
                try:
                    prediction = model.predict(input_data_scaled)
                    predictions[target_name] = prediction[0]
                except Exception as e:
                    logger.error(f"Error making prediction with model {target_name}: {str(e)}")
                    predictions[target_name] = "Error"

        except Exception as e:
            logger.error(f"Error in prediction route: {str(e)}")
            flash(f"An error occurred: {str(e)}", 'danger')

    return render_template('predict.html', predictions=predictions)
if __name__ == '__main__':
    app.run(debug=True)
