from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    return render_template('index.html')


# =========================
# MANUAL PREDICTION
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    try:

        amount = float(request.form['amount'])
        time = float(request.form['time'])
        transaction_type = int(request.form['transaction_type'])
        international = int(request.form['international'])
        merchant_risk = int(request.form['merchant_risk'])

        # Create 30-feature vector
        features = [0] * 30

        # Fill important features
        features[0] = time
        features[1] = amount / 100
        features[2] = transaction_type
        features[3] = international
        features[4] = merchant_risk

        # Convert to array
        input_array = np.array(features).reshape(1, -1)

        # Scale input
        scaled_input = scaler.transform(input_array)

        # ML prediction
        prediction = model.predict(scaled_input)[0]

        # Extra fraud scoring logic
        fraud_score = (
            amount * 0.02
            + international * 40
            + merchant_risk * 35
        )

        if fraud_score > 60 or prediction == 1:

            result = "⚠ Fraudulent Transaction Detected"

        else:

            result = "✅ Genuine Transaction"

        return render_template(
            'index.html',
            prediction_text=result
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


# =========================
# CSV UPLOAD PREDICTION
# =========================
@app.route('/upload', methods=['POST'])
def upload():

    try:

        file = request.files['file']

        if file.filename == '':

            return render_template(
                'index.html',
                prediction_text="No file selected"
            )

        # Read CSV
        df = pd.read_csv(file)

        # Remove Class column if exists
        if 'Class' in df.columns:

            X = df.drop('Class', axis=1)

        else:

            X = df

        # Scale data
        scaled_data = scaler.transform(X)

        # Predict
        predictions = model.predict(scaled_data)

        # Count frauds
        fraud_count = np.sum(predictions == 1)
        genuine_count = np.sum(predictions == 0)

        result = f"""
        CSV Analysis Complete

        Total Transactions: {len(predictions)}

        Genuine Transactions: {genuine_count}

        Fraudulent Transactions: {fraud_count}
        """

        return render_template(
            'index.html',
            prediction_text=result
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"CSV Error: {str(e)}"
        )


# =========================
# RUN APP
# =========================
if __name__ == '__main__':

    app.run(debug=True)