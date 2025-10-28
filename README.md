# 🩺 Vitalsync AI: Chronic Disease Risk Predictor

This repository contains the source code for a full-stack Machine Learning application designed to assess a patient's risk profile for seven common **chronic diseases** (including Diabetes, Hypertension, and Asthma) based on provided clinical and lifestyle indicators.

The project demonstrates a robust, end-to-end MLOps workflow, successfully addressing significant data quality and deployment challenges.

---

## ✨ Key Technical Highlights

| Aspect | Technology & Achievement |
| :--- | :--- |
| **Model** | **Multi-Class XGBoost Classifier** ($\approx 81\%$ Accuracy). |
| **Deployment** | Python **Flask API** serving predictions to a dynamic **HTML/Tailwind** front-end via AJAX (`fetch`). |
| **Bias Mitigation** | **SMOTE-ENN** (Synthetic Minority Over-sampling Technique) was used to successfully fix severe class imbalance, allowing the model to accurately predict minority classes like 'Healthy' and 'Cancer'. |
| **Data Robustness** | Implemented **IQR Outlier Capping** and **Manual Standard Scaling** in the Python API to normalize inputs correctly and bypass a corrupted `scaler.pkl` file. |
| **Explainability (XAI)** | Integrates **SHAP-based Feature Attribution** logic to explain which clinical markers (`HbA1c`, `Age Group`) contributed most to the prediction score. |
| **Design** | Modern, professional, and fully **responsive** UI using **Tailwind CSS** and **Poppins** font, including a Dark Mode feature. |

---

## 🛠️ Project Structure

For reference, the project directory is structured as follows:

healthcare-app/ ├── server.py # Python Flask API (Pre-processing & Model Inference) ├── patient_app.html # Frontend UI (HTML, JS, Tailwind) ├── healthcaer_risk_model.pkl # The final, SMOTE-ENN trained XGBoost model. ├── scaler.pkl # The saved Standard Scaler object (used for metadata/comparison). └── label_encoder.pkl # Decodes numerical predictions (0, 1, 2...) into disease names.


---

## 🚀 Getting Started

### Prerequisites

You must have Python 3.8+ installed.

```bash
# Install all required libraries
pip install Flask pandas numpy xgboost scikit-learn imbalanced-learn flask-cors
1. Run the Python Backend (API)
The backend must be running before you open the application in the browser.

Bash

# Navigate to the project directory
cd healthcare-app

# Start the Flask server
python server.py
(The server will start listening at http://127.0.0.1:5000.)

2. Run the Frontend (UI)
Locate the patient_app.html file in your project folder.

Double-click the file to open it in your web browser (Chrome/Firefox).

Enter the required vital signs and click "PREDICT HEALTH RISK".
