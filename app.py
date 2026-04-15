# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
import warnings
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Load and preprocess data with error handling
try:
    df = pd.read_csv("ADXL345_SensorData.csv")  # Replace with your file path if different
    if df.empty or 'Error_found' not in df.columns or df[['X-direction', 'Y-direction', 'Z-direction']].isnull().all().any():
        raise ValueError("Dataset is empty or missing required columns (X-direction, Y-direction, Z-direction, Error_found)")
except FileNotFoundError:
    raise FileNotFoundError("Error: 'ADXL345_SensorData.csv' not found. Please provide the correct file path.")
except Exception as e:
    raise Exception(f"Error loading dataset: {str(e)}")

# Encode target variable
le = LabelEncoder()
df['Error_found'] = df['Error_found'].astype(str).replace({'yes': 1, 'no': 0})  # Ensure 'yes'/'no' mapping
df['Error_found'] = le.fit_transform(df['Error_found'])  # 'no' = 0, 'yes' = 1

# Feature and target split
X = df[['X-direction', 'Y-direction', 'Z-direction']]  # Explicitly select features
y = df['Error_found']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models with probability support
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}

# Train and evaluate models
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    except Exception as e:
        results[name] = 0.0
        print(f"Error training {name}: {str(e)}")

# Train and evaluate Voting Ensemble
try:
    voting_model = VotingClassifier(estimators=[(name, model) for name, model in models.items()], voting='soft')
    voting_model.fit(X_train, y_train)
    y_pred_voting = voting_model.predict(X_test)
    results['Voting Ensemble'] = accuracy_score(y_test, y_pred_voting)
except Exception as e:
    results['Voting Ensemble'] = 0.0
    print(f"Error training Voting Ensemble: {str(e)}")

# Identify best model
best_model_name = max(results, key=results.get)
best_model = voting_model if best_model_name == 'Voting Ensemble' else models[best_model_name]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        x_input = float(data['x'])
        y_input = float(data['y'])
        z_input = float(data['z'])

        custom_input = np.array([[x_input, y_input, z_input]])
        custom_input_scaled = scaler.transform(custom_input)
        custom_pred = best_model.predict(custom_input_scaled)
        prediction = le.inverse_transform(custom_pred)[0].upper()

        confidence = 100
        if hasattr(best_model, 'predict_proba'):
            prob = best_model.predict_proba(custom_input_scaled)[0]
            confidence = prob[custom_pred[0]] * 100

        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'input_values': [x_input, y_input, z_input]
        })
    except ValueError as ve:
        return jsonify({'success': False, 'error': f'Invalid input: {str(ve)}'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)