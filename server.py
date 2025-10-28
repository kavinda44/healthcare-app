import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS 
import sys 

app = Flask(__name__)
CORS(app) 


CAP_BOUNDS = {
    'Glucose': {'lower': 75, 'upper': 250}, 'Blood Pressure': {'lower': 90, 'upper': 170},
    'BMI': {'lower': 18, 'upper': 40}, 'Cholesterol': {'lower': 150, 'upper': 300},
    'HbA1c': {'lower': 4.0, 'upper': 10.0}, 'Sleep Hours': {'lower': 4.0, 'upper': 10.0},
    'Physical Activity': {'lower': 0, 'upper': 450}, 'Oxygen Saturation': {'lower': 90.0, 'upper': 100.0},
    'LengthOfStay': {'lower': 1, 'upper': 10}, 'Triglycerides': {'lower': 50, 'upper': 350}, 
    'Diet Score': {'lower': 1, 'upper': 10}, 'Stress Level': {'lower': 1, 'upper': 10}
}


SAVED_SCALING_STATS = {
    'Glucose': {'mean': 125.0, 'std': 30.0}, 
    'Blood Pressure': {'mean': 128.0, 'std': 18.0},
    'BMI': {'mean': 28.0, 'std': 5.0}, 
    'Cholesterol': {'mean': 205.0, 'std': 40.0},
    'HbA1c': {'mean': 6.2, 'std': 1.2}, 
    'Sleep Hours': {'mean': 7.2, 'std': 1.5},
    'Physical Activity': {'mean': 180.0, 'std': 100.0}, 
    'Oxygen Saturation': {'mean': 97.5, 'std': 0.0}, 
    'LengthOfStay': {'mean': 3.5, 'std': 2.0}, 
    'Triglycerides': {'mean': 150.0, 'std': 50.0},
    'Diet Score': {'mean': 5.0, 'std': 2.0}, 
    'Stress Level': {'mean': 5.0, 'std': 2.0}
}


def cap_outliers_server(df, bounds):
    """Applies capping to continuous features based on defined boundaries."""
    for col, limits in bounds.items():
        if col in df.columns:
            df[col] = np.where(df[col] > limits['upper'], limits['upper'], df[col])
            df[col] = np.where(df[col] < limits['lower'], limits['lower'], df[col])
    return df


try:
    with open('healthcare_risk_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f) 
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    SCALING_COLS_ORDERED = [
        'Glucose', 'Blood Pressure', 'BMI', 'Oxygen Saturation', 'LengthOfStay', 
        'Cholesterol', 'Triglycerides', 'HbA1c', 'Physical Activity', 'Diet Score', 
        'Stress Level', 'Sleep Hours' 
    ]
    
    MODEL_FEATURES = [
        'Glucose', 'Blood Pressure', 'BMI', 'Oxygen Saturation', 'LengthOfStay', 
        'Cholesterol', 'Triglycerides', 'HbA1c', 'Physical Activity', 'Diet Score', 
        'Stress Level', 'Sleep Hours', 'Smoking', 'Alcohol', 'Family History', 
        'Gender_Male', 'AgeGroup_Age_Adult', 'AgeGroup_Age_Middle_Aged', 'AgeGroup_Age_Senior'
    ]

    print("Model and assets loaded successfully!")
except Exception as e:
    print(f"Error loading assets. Please check file names and directory. Error: {e}")
    sys.exit(1) 
    

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
  
    key_mapping = {
        'Age': 'Age', 'Gender': 'Gender', 'Glucose': 'Glucose', 
        'BloodPressure': 'Blood Pressure', 'BMI': 'BMI', 'Cholesterol': 'Cholesterol', 
        'HbA1c': 'HbA1c', 'SleepHours': 'Sleep Hours', 'PhysicalActivity': 'Physical Activity', 
        'Smoking': 'Smoking', 'Alcohol': 'Alcohol', 'FamilyHistory': 'Family History',
        'OxygenSaturation': 'Oxygen Saturation', 'LengthOfStay': 'LengthOfStay', 
        'Triglycerides': 'Triglycerides', 'DietScore': 'Diet Score', 'StressLevel': 'Stress Level'
    }
    
    input_data = {
        'Oxygen Saturation': 97.5, 'LengthOfStay': 2, 'Triglycerides': 150, 
        'Diet Score': 5, 'Stress Level': 5
    }
    
    for html_key, py_column_name in key_mapping.items():
        if html_key in data:
            try:
                value = float(data[html_key]) if data[html_key] not in ['Male', 'Female'] else data[html_key]
            except ValueError:
                value = data[html_key] 
            input_data[py_column_name] = value

    raw_feature_names_to_use = [
        'Age', 'Gender', 'Glucose', 'Blood Pressure', 'BMI', 'Cholesterol', 'HbA1c', 
        'Sleep Hours', 'Physical Activity', 'Smoking', 'Alcohol', 'Family History', 
        'Oxygen Saturation', 'LengthOfStay', 'Triglycerides', 'Diet Score', 'Stress Level'
    ]
    raw_df_data = {k: input_data.get(k) for k in raw_feature_names_to_use}
    raw_df = pd.DataFrame([raw_df_data])
    
  
    
    
    raw_df = cap_outliers_server(raw_df, CAP_BOUNDS)

   
    raw_df['Age'] = pd.to_numeric(raw_df['Age'], errors='coerce')
    bins = [0, 18, 40, 65, np.inf]
    labels_raw = ['AgeGroup_Child', 'AgeGroup_Age_Adult', 'AgeGroup_Age_Middle_Aged', 'AgeGroup_Age_Senior'] 
    raw_df['AgeGroup'] = pd.cut(raw_df['Age'], bins=bins, labels=labels_raw, right=False)
    
    
    processed_df = raw_df.drop('Age', axis=1).copy()
    processed_df = pd.get_dummies(processed_df, columns=['Gender', 'AgeGroup'], drop_first=True, dtype=int)
    
    
    condition_dummies = [col for col in MODEL_FEATURES if col.startswith('Medical Condition_')]
    for col in condition_dummies:
        processed_df[col] = 0
    
    final_X = processed_df.reindex(columns=MODEL_FEATURES, fill_value=0)
    
   
    try:
        for col in SCALING_COLS_ORDERED:
            mean = SAVED_SCALING_STATS[col]['mean']
            std = SAVED_SCALING_STATS[col]['std']
            
            if std == 0:
               
                final_X[col] = 0.0
            else:
               
                final_X[col] = (final_X[col] - mean) / std 
        
    except Exception as scale_e:
        print(f"Manual Standard Scaling Failed. Error: {scale_e}")
        return jsonify({"error": "Prediction processing failed: Scaling Error."}), 500

    
    X_predict = final_X[MODEL_FEATURES].values 
    
    prediction_encoded = model.predict(X_predict)[0]
    prediction_proba = model.predict_proba(X_predict)[0].tolist()
    
  
    predicted_condition = label_encoder.inverse_transform([prediction_encoded])[0]
    
    
    explanations = []
    if predicted_condition == 'Hypertension':
        explanations = [{'feature': 'High Blood Pressure', 'risk': 0.35}, {'feature': 'Age Group (Senior)', 'risk': 0.15}]
    elif predicted_condition == 'Diabetes':
        explanations = [{'feature': 'High HbA1c', 'risk': 0.40}, {'feature': 'High Glucose', 'risk': 0.25}]
    elif predicted_condition == 'Healthy':
        explanations = [{'feature': 'Optimal Vitals', 'risk': -0.30}, {'feature': 'Regular Activity', 'risk': -0.15}]
    else:
        explanations = [{'feature': predicted_condition, 'risk': 0.20}, {'feature': 'Check Lifestyle', 'risk': -0.10}]
        
    max_proba = max(prediction_proba)
    
    return jsonify({
        'predictedCondition': predicted_condition,
        'probabilities': prediction_proba,
        'confidence': f"{max_proba * 100:.2f}%",
        'explanations': explanations,
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)