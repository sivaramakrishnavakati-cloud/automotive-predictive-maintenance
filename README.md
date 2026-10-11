# AI-Powered Engine Condition Classification

## Overview

A machine learning application that analyzes engine sensor measurements
and predicts an engine-condition class using trained classification models.

This project demonstrates data exploration, machine learning,
model evaluation, feature importance, API development, and dashboard creation.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Logistic Regression
- Random Forest
- Gradient Boosting
- Streamlit
- FastAPI
- Pytest

## Project Features

- Dataset exploration and quality checks
- Training and comparison of three ML models
- Feature importance visualization
- Interactive Streamlit dashboard
- FastAPI prediction endpoint
- Automated API tests

## Project Structure

```text
automotive-predictive-maintenance/
├── api.py
├── app.py
├── data/
├── models/
├── reports/
├── src/
│   ├── explore_data.py
│   ├── feature_analysis.py
│   └── train_model.py
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md