# Credit Card Default Risk Prediction API (Taiwan)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack & Tools](#tech-stack--tools)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Model Training Pipeline](#model-training-pipeline)
- [API Service & Deployment](#api-service--deployment)
- [Running Locally](#running-locally)
- [Deploying on Render (Cloud)](#deploying-on-render-cloud)
- [Testing & CI/CD](#testing--cicd)
- [API Usage Guide](#api-usage-guide)
- [Project Impact & Benefits](#project-impact--benefits)
- [License](#license)
- [Contact](#contact)

---

## 📝 Project Overview

This project delivers a **machine learning solution for predicting credit card default risk**, covering all stages from data processing, feature engineering, model training to scalable API deployment.  
The final product is a **FastAPI RESTful web service**, which supports real-time inference via a public cloud endpoint, containerized via Docker, and auto-deployed on Render.

---

## 🌟 Features

- **Modularized code for data preparation and feature engineering**
- **High-performance ML model (LightGBM + EasyEnsemble)**
- **Automated training script with MLflow for experiment tracking (local/dev)**
- **Production-ready FastAPI REST API with Pydantic input validation**
- **One-click deployment on Render (Cloud PaaS with free tier)**
- **CI/CD integration (GitHub Actions): automated testing, linting, deployment**
- **API Key authentication supported for secure access**

---

## 🛠️ Tech Stack & Tools

- **Language/Frameworks:** Python 3.11, FastAPI, Pydantic, Uvicorn
- **Machine Learning:** LightGBM, scikit-learn, imblearn
- **Model Packaging:** joblib, MLflow (for development)
- **Data Processing:** Pandas, NumPy
- **API Deployment:** Docker, Render (free public cloud)
- **Experiment Tracking:** MLflow (local, optional)
- **Automation:** GitHub Actions (CI/CD), pytest (unit testing), flake8 (static analysis)
- **Version Control:** Git, GitHub

---

## 📂 Repository Structure

```text
credit-default-taiwan/
│
├── notebooks/                  # Jupyter notebooks for experiments
│   └── Credit_Default_Risk.ipynb
├── src/                        # Data and feature engineering modules
│   ├── __init__.py
│   ├── data_prep.py
│   ├── feature_eng.py
│   ├── model_utils.py
│   └── train.py
├── artifacts/                  # Trained model artifacts
│   └── model.joblib
├── api/                        # FastAPI service code
│   ├── app.py
│   └── schema.py
├── test/                       # Unit tests
│   └── test_app.py
├── requirements.txt            # Main project dependencies
├── requirements-dev.txt        # Dev/test dependencies
├── Dockerfile                  # Docker build recipe
├── .dockerignore
├── .gitignore
├── .flake8                     # flake8 configuration
├── README.md                   # (this file)
└── LICENSE                     # License info
```

---

## 📑 Dataset

- **Source:** [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **Samples:** 30,000 entries
- **Features:** 23 columns (including demographics, balance, billing, and payment records)
- **Goal:** Predict whether a client will default on payment next month

---

## 🚀 Model Training Pipeline

1. **Data Cleaning & Preparation** (`src/data_prep.py`)
   - Handling missing values
   - Categorical encoding
   - Outlier and extreme value treatment

2. **Feature Engineering** (`src/feature_eng.py`)
   - Generating derived features
   - Standardization, binning, and other transformations

3. **Model Training** (`src/model_utils.py`, `src/train.py`)
   - LightGBM with EasyEnsemble for imbalanced classification
   - Evaluation using accuracy, AUC, recall, and other metrics
   - Exporting the final pipeline to `artifacts/model.joblib`

4. **MLflow Experiment Tracking**
   - *Optional*: Only for local development and experiment tracking

---

## 🌐 API Service & Deployment

- **Framework**: FastAPI
- **Inference**: Loads model directly from `artifacts/model.joblib`
- **Input Validation**: Handled with Pydantic schema
- **Security**: API Key authentication supported (customizable)

### 🔌 Main Endpoints

- `GET /`  
  - Health check endpoint to confirm the service is live.

- `POST /predict`  
  - Accepts a batch of client records in JSON format.  
  - Returns prediction results with probability scores and binary labels.

---

## 🖥️ Running Locally

### 1. Install dependencies

```bash
git clone https://github.com/hungchenhsu/credit-default-taiwan.git
cd credit-default-taiwan
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
# Output: artifacts/model.joblib
```

### 3. Launch the API server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 4. Example API call

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "records": [{
          "LIMIT_BAL": 200000,
          "SEX": 2,
          "EDUCATION": 2,
          "MARRIAGE": 1,
          "AGE": 29,
          "PAY_0": 1,  "PAY_2": 0, "PAY_3": 0,
          "PAY_4": 0,  "PAY_5": 0, "PAY_6": 0,
          "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
          "BILL_AMT4": 0,    "BILL_AMT5": 0,    "BILL_AMT6": 0,
          "PAY_AMT1": 0, "PAY_AMT2": 1000, "PAY_AMT3": 1000,
          "PAY_AMT4": 1000, "PAY_AMT5": 0, "PAY_AMT6": 0
        }]
      }'
```


