# Aircraft Engine RUL Prediction

## Project Overview

This project predicts the Remaining Useful Life (RUL) of aircraft engines using Machine Learning.

The system predicts:
- Remaining Useful Life in cycles
- Engine condition
- Risk level
- Maintenance recommendations

## Dataset

NASA C-MAPSS FD001 simulated turbofan engine dataset.

## Machine Learning Model

Random Forest Regressor

## Model Performance

Official test-set results:

- MAE: 19.86 cycles
- RMSE: 27.02 cycles
- R2 Score: 0.5771

Validation results:

- MAE: 23.94 cycles
- RMSE: 31.58 cycles
- R2 Score: 0.7686

## Condition Monitoring

Project-defined thresholds:

- RUL greater than 50 cycles: Healthy / Low Risk
- RUL between 20 and 50 cycles: Warning / Medium Risk
- RUL below 20 cycles: Critical / High Risk

## Maintenance Recommendations

Healthy:
Continue routine monitoring and scheduled maintenance.

Warning:
Increase monitoring frequency and schedule preventive maintenance.

Critical:
Immediate inspection recommended and maintenance should be prioritized.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Random Forest, Streamlit and Joblib.

## Deployment

The trained Random Forest model is deployed using Streamlit for interactive RUL prediction and engine condition monitoring.

## Disclaimer

This project uses simulated aircraft engine data for educational and predictive maintenance analysis. It should not be used for real-world aircraft maintenance or safety decisions.


## Interactive 3D Explorer

The Streamlit app includes a procedural Three.js aircraft model with selectable component meshes, hover tooltips, component search, camera presets, and a component inspector. The procedural model is a visualization/demo, not a certified aircraft model.

### Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The 3D renderer loads Three.js from a CDN, so an internet connection is required to display the interactive model. The app can still show the project interface without the 3D libraries loading.

### Model file note

The supplied archive contains `rul_features.pkl` and saved prediction records, but does not contain `aircraft_rul_random_forest.pkl`. Live RUL prediction will remain disabled until the trained model file is placed beside `app.py`. Saved CSV records are shown as sample results only, not treated as live predictions.

The aircraft is procedurally constructed for the demonstration and has independently selectable meshes. It is not a detailed engineering model and must not be used for real maintenance decisions.
