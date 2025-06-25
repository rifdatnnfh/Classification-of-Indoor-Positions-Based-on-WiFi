# Prototype Lengkap: Prediksi Lokasi Berdasarkan Wi-Fi Snapshot
# Backend dengan Flask dan model SVM

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- Model Training ---
def train_svm_model():
    df = pd.read_csv('Dataset_Model_Preprocessed.csv')
    df.fillna(-100, inplace=True)  # Anggap -100 sebagai AP tidak terdeteksi

    X = df.drop('spot', axis=1)
    y = df['spot']

    model = SVC(kernel='rbf', C=1, gamma='scale')
    model.fit(X, y)

    return model, X.columns

# Load atau latih model
MODEL_PATH = 'model_svm.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    ap_columns = joblib.load('ap_columns.pkl')
else:
    model, ap_columns = train_svm_model()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(ap_columns, 'ap_columns.pkl')

# --- Web Routes ---
@app.route('/')
def index():
    ap_names = [col for col in ap_columns if col.startswith('ap_')]
    return render_template('index.html', ap_names=ap_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = {ap_name: -100 for ap_name in ap_columns}

        for ap_name in ap_columns:
            rssi_val = request.form.get(ap_name)
            if rssi_val:
                try:
                    input_data[ap_name] = float(rssi_val)
                except ValueError:
                    input_data[ap_name] = -100

        input_df = pd.DataFrame([input_data])[ap_columns]
        prediction = model.predict(input_df)
        predicted_floor = prediction[0]

        return jsonify({'predicted_floor': predicted_floor})

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
