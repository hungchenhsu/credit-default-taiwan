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
