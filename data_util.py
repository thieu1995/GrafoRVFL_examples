#!/usr/bin/env python
# Created by "Thieu" at 17:55, 30/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, fetch_openml, fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Configuration for train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Dictionary to store dataset information
datasets = {}

# Helper function to standardize datasets and encode categorical target if needed
def preprocess_data(X, y):
    # Convert X to a DataFrame if it's a numpy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # Ensure y is a Series (in case it's passed as numpy array)
    y = pd.Series(y)

    # Combine X and y into a single DataFrame to drop NaN rows in both
    data = X.copy()
    data['target'] = y

    # Remove rows with NaN values
    data = data.dropna()

    # Separate X and y after dropping NaNs
    X = data.drop(columns=['target'])
    y = data['target']

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=['number']).columns

    # Define transformers for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Apply transformations
    X_processed = preprocessor.fit_transform(X)

    # Check if target is categorical and encode it if necessary
    y = y.values
    if y.dtype == 'object' or y.dtype.name == 'category':  # Object type, usually indicating non-numeric labels
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test

##################################################

def get_iris():
    df = load_iris(as_frame=True)
    return preprocess_data(df.data, df.target)


def get_breast_cancer():
    df = load_breast_cancer(as_frame=True)
    return preprocess_data(df.data, df.target)


def get_digits():
    df = load_digits(as_frame=True)
    return preprocess_data(df.data, df.target)


def get_wine():
    df = fetch_openml(name='wine', version=1, as_frame=True, parser="auto")
    return preprocess_data(df.data, df.target)


def get_phoneme():
    df = fetch_openml(name='phoneme', version=1, as_frame=True, parser="auto")
    return preprocess_data(df.data, df.target.astype(int))


def get_waveform():
    df = fetch_openml(name='waveform-5000', version=1, as_frame=True, parser="auto")
    return preprocess_data(df.data, df.target.astype(int))


def get_magic_telescope():
    df = fetch_openml(name='MagicTelescope', version=2, as_frame=True, parser="auto")
    return preprocess_data(df.data, df.target)


def get_diabetes():
    df = load_diabetes(as_frame=True)
    return preprocess_data(df.data, df.target)  # Categorical y


def get_boston_housing():
    df = fetch_openml(name='boston', version=1, as_frame=True, parser="auto")
    return preprocess_data(df.data, df.target)


def get_california_housing():
    df = fetch_california_housing()
    return preprocess_data(df.data, df.target)


# # 1. Iris dataset (Classification)
# data = load_iris(as_frame=True)
# datasets['iris'] = preprocess_data(data.data, data.target)
#
# # 2. Breast Cancer dataset (Classification)
# data = load_breast_cancer(as_frame=True)
# datasets['breast_cancer'] = preprocess_data(data.data, data.target)
#
# # 3. Digits dataset (Classification)
# data = load_digits(as_frame=True)
# datasets['digits'] = preprocess_data(data.data, data.target)
#
# # 4. Wine dataset (Classification)
# data = fetch_openml(name='wine', version=1, as_frame=True, parser="auto")
# datasets['wine'] = preprocess_data(data.data, data.target)
#
# # 5. Phoneme dataset (Classification)
# data = fetch_openml(name='phoneme', version=1, as_frame=True, parser="auto")
# datasets['phoneme'] = preprocess_data(data.data, data.target.astype(int))
#
# # 6. Waveform dataset (Classification)
# data = fetch_openml(name='waveform-5000', version=1, as_frame=True, parser="auto")
# datasets['waveform'] = preprocess_data(data.data, data.target.astype(int))
#
# # 7. Magic Gamma Telescope dataset (Classification)
# data = fetch_openml(name='MagicTelescope', version=2, as_frame=True, parser="auto")
# datasets['magic_gamma'] = preprocess_data(data.data, data.target)
#
# # # 8. Adult Income dataset (Classification)
# # data = fetch_openml(name='adult', version=2, as_frame=True, parser="auto")
# # # Convert categorical target into numerical values
# # data.target = data.target.map({'<=50K': 0, '>50K': 1})
# # datasets['adult_income'] = preprocess_data(data.data, data.target)
#
# # 8. Diabetes dataset (Regression)
# data = fetch_openml(name='diabetes', version=1, as_frame=True, parser="auto")
# datasets['diabetes'] = preprocess_data(data.data, data.target)      # Categorical y
#
# # 9. Boston Housing dataset (Regression)
# data = fetch_openml(name='boston', version=1, as_frame=True, parser="auto")
# datasets['boston'] = preprocess_data(data.data, data.target)
#
# # 10. California Housing Dataset
# data = fetch_california_housing()
# datasets['california'] = preprocess_data(data.data, data.target)
#
# # # 11. Ames Housing Dataset (OpenML)
# # data = fetch_openml("house_prices", version=1, as_frame=True, parser="auto")
# # X_ames = data.data.select_dtypes(include=[np.number]).fillna(0)  # Select numerical and fill NAs
# # X_ames.drop(["Id", "YearBuilt", "YearRemodAdd", "MasVnr...PorchSF", ], axis=1, inplace=True)
# # datasets['ames'] = preprocess_data(X_ames, data.target)
# # # 'Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
# # #        'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnr...PorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
# # #        'MiscVal', 'MoSold', 'YrSold'
#
#
# # Check the datasets
# for name, (X_train, X_test, y_train, y_test) in datasets.items():
#     print(f"{name.capitalize()} Dataset:")
#     print(f"  Training data shape: {X_train.shape}")
#     print(f"  Test data shape: {X_test.shape}")
#     print(f"  Training target shape: {y_train.shape}")
#     print(f"  Test target shape: {y_test.shape}\n")
