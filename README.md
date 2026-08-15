
# Credit Card Fraud Detection Using Machine Learning

A machine learning-based web application that detects potentially fraudulent credit card transactions. The system uses Python and Scikit-learn for machine learning and Flask to integrate the trained model with a user-friendly web interface.

=> Features

* Fraud vs. genuine transaction classification
* Manual transaction analysis
* CSV file upload for analyzing multiple transactions
* Fraud risk assessment
* Simple and responsive web interface
* Machine learning model integration using Flask
* Data processing using Pandas and NumPy

=> Technologies Used

* **Python** – Core programming language
* **Flask** – Backend web framework
* **Scikit-learn** – Machine learning
* **Pandas** – Data processing and CSV handling
* **NumPy** – Numerical operations
* **Joblib** – Saving and loading the trained model
* **HTML & CSS** – Frontend interface

=> Dataset

The project uses the Credit Card Fraud Detection dataset. The dataset contains transaction features such as `Time`, `Amount`, anonymized PCA-transformed features (`V1`–`V28`), and `Class`.

* `0` → Genuine transaction
* `1` → Fraudulent transaction

=> How It Works
Credit Card Dataset
        ↓
Data Preprocessing
        ↓
Feature Scaling
        ↓
Machine Learning Model
        ↓
Saved Model
        ↓
Flask Backend
        ↓
Web Interface
        ↓
Fraud / Genuine Prediction

For CSV analysis, the application reads the uploaded transaction data, applies the same preprocessing used during training, sends the data to the trained model, and displays the prediction results.

