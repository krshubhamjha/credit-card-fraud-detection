import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_model_exists():
    assert os.path.exists('model/xgb_model.pkl')

def test_scaler_exists():
    assert os.path.exists('model/scaler.pkl')

def test_feature_names_exists():
    assert os.path.exists('model/feature_names.json')

def test_model_loads():
    with open('model/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    assert model is not None

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

def test_app_exists():
    assert os.path.exists('app.py')