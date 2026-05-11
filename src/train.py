import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
import json
import os
import sys
from datetime import datetime


def validate_data(df, label="data"):
    print(f"validating {label}...")
    errors = []

    expected_columns = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
        'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
        'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23',
        'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'
    ]

    # check all columns are present
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")

    # check no missing values
    if df.isnull().sum().sum() > 0:
        errors.append("missing values found")

    # check class column has only 0 and 1
    invalid_classes = set(df['Class'].unique()) - {0, 1}
    if invalid_classes:
        errors.append(f"invalid class values: {invalid_classes}")

    # check minimum rows
    if len(df) < 1000:
        errors.append(f"too few rows: {len(df)}")

    # check fraud cases exist
    if df['Class'].sum() == 0:
        errors.append("no fraud cases found")

    # check no negative amounts
    if df['Amount'].min() < 0:
        errors.append("negative amounts found")

    if errors:
        print("validation failed!")
        for e in errors:
            print(f"  error: {e}")
        sys.exit(1)

    print(f"validation passed for {label}")


def append_new_data():
    new_file  = 'data/new_data/new_transactions.csv'
    main_file = 'data/creditcard.csv'
    log_file  = 'data/append_log.csv'

    # if no new data just skip
    if not os.path.exists(new_file):
        print("no new data found, using existing data")
        return

    old_df = pd.read_csv(main_file)
    new_df = pd.read_csv(new_file)

    print(f"existing rows: {len(old_df)}")
    print(f"new rows: {len(new_df)}")

    # combine and remove duplicates
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates()

    new_rows = len(combined) - len(old_df)

    if new_rows <= 0:
        print("no new rows to add, skipping")
        return

    # save combined data
    combined.to_csv(main_file, index=False)
    print(f"appended {new_rows} new rows")
    print(f"total rows now: {len(combined)}")

    # save log
    log_entry = pd.DataFrame([{
        'date':        datetime.now().strftime('%Y-%m-%d %H:%M'),
        'rows_added':  new_rows,
        'total_rows':  len(combined),
        'fraud_cases': int(combined['Class'].sum())
    }])

    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        log_df = pd.concat([log_df, log_entry], ignore_index=True)
    else:
        log_df = log_entry

    log_df.to_csv(log_file, index=False)
    print("log updated")


def load_data():
    df = pd.read_csv('data/creditcard.csv')
    print(f"data loaded: {len(df)} rows")
    print(f"fraud cases: {df['Class'].sum()}")
    return df


def feature_engineering(df):
    scaler = StandardScaler()

    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_Scaled']   = scaler.fit_transform(df[['Time']])
    df['Hour']          = (df['Time'] / 3600) % 24

    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # save processed data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/processed_data.csv', index=False)
    print("feature engineering done")

    return df, scaler


def train_model(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # apply smote
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"after smote: {y_train_smote.sum()} fraud cases")

    # train model
    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_smote, y_train_smote)

    # evaluate
    pred   = model.predict(X_test)
    recall = recall_score(y_test, pred)
    f1     = f1_score(y_test, pred)

    print(f"recall: {recall:.4f}")
    print(f"f1 score: {f1:.4f}")
    print(classification_report(y_test, pred))

    return model, X_test, y_test, recall, X.columns.tolist()


def compare_and_save(new_model, scaler, feature_names, X_test, y_test, new_recall):
    if os.path.exists('model/xgb_model.pkl'):
        with open('model/xgb_model.pkl', 'rb') as f:
            old_model = pickle.load(f)

        old_pred   = old_model.predict(X_test)
        old_recall = recall_score(y_test, old_pred)

        print(f"old model recall: {old_recall:.4f}")
        print(f"new model recall: {new_recall:.4f}")

        if new_recall > old_recall:
            print("new model is better, saving")
            save_model(new_model, scaler, feature_names)
        else:
            print("old model is better, keeping old model")
    else:
        print("no old model found, saving new model")
        save_model(new_model, scaler, feature_names)


def save_model(model, scaler, feature_names):
    os.makedirs('model', exist_ok=True)

    with open('model/xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('model/feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    print("model saved")


if __name__ == '__main__':

    # validate new data first before appending
    if os.path.exists('data/new_data/new_transactions.csv'):
        new_df = pd.read_csv('data/new_data/new_transactions.csv')
        validate_data(new_df, label="new data")
        append_new_data()
    else:
        print("no new data, skipping append")

    # load combined data
    df = load_data()

    # validate combined data
    validate_data(df, label="combined data")

    # feature engineering
    df, scaler = feature_engineering(df)

    # train model
    model, X_test, y_test, recall, feature_names = train_model(df)

    # compare and save
    compare_and_save(model, scaler, feature_names, X_test, y_test, recall)

    print("pipeline done!")