import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
import pytest

# add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# test 1 - model file exists
def test_model_exists():
    assert os.path.exists('model/xgb_model.pkl')


# test 2 - scaler file exists
def test_scaler_exists():
    assert os.path.exists('model/scaler.pkl')


# test 3 - feature names file exists
def test_feature_names_exists():
    assert os.path.exists('model/feature_names.json')


# test 4 - model loads without error
def test_model_loads():
    with open('model/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    assert model is not None


# test 5 - model predicts 0 or 1
def test_model_predicts():
    with open('model/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/feature_names.json', 'r') as f:
        feature_names = json.load(f)

    dummy_input = pd.DataFrame(
        [np.zeros(len(feature_names))],
        columns=feature_names
    )

    prediction = model.predict(dummy_input)
    assert prediction[0] in [0, 1]


# test 6 - processed data exists
def test_processed_data_exists():
    assert os.path.exists('data/processed/processed_data.csv')


# test 7 - processed data has correct columns
def test_processed_data_columns():
    df = pd.read_csv('data/processed/processed_data.csv')
    assert 'Class' in df.columns
    assert 'Amount_Scaled' in df.columns
    assert 'Hour' in df.columns


# test 8 - processed data has no missing values
def test_no_missing_values():
    df = pd.read_csv('data/processed/processed_data.csv')
    assert df.isnull().sum().sum() == 0


# test 9 - class column has only 0 and 1
def test_class_values():
    df = pd.read_csv('data/processed/processed_data.csv')
    assert set(df['Class'].unique()).issubset({0, 1})


# test 10 - app.py exists
def test_app_exists():
    assert os.path.exists('app.py')