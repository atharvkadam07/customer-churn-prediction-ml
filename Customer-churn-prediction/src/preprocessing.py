"""
Customer Churn Prediction
Data Preprocessing Pipeline

This module:
1. Loads the raw dataset
2. Cleans the data
3. Handles missing values
4. Separates features and target
5. Identifies numerical and categorical features
6. Builds a preprocessing pipeline
7. Splits data into training and testing sets
8. Saves the preprocessing pipeline
"""

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# PROJECT PATHS
# ============================================================

# Get the folder containing this file:
# Customer-churn-prediction/src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up:
# Customer-churn-prediction
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Dataset path:
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "customer_churn.csv"
)

# Models directory:
MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

# Saved preprocessing pipeline:
PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the raw customer churn dataset.
    """

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATASET LOADED")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):
    """
    Clean the raw dataset.
    """

    df = df.copy()

    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    # --------------------------------------------------------
    # Remove customer ID
    # --------------------------------------------------------

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

        print("Removed: customerID")

    # --------------------------------------------------------
    # Convert TotalCharges to numeric
    # --------------------------------------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Check missing TotalCharges
    # --------------------------------------------------------

    missing_total_charges = df["TotalCharges"].isna().sum()

    print(
        f"Missing TotalCharges after conversion: "
        f"{missing_total_charges}"
    )

    # --------------------------------------------------------
    # Convert target variable
    # --------------------------------------------------------

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    print("Converted Churn: Yes = 1, No = 0")

    return df


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

def split_features_target(df):
    """
    Separate input features (X) and target variable (y).
    """

    X = df.drop(columns=["Churn"])

    y = df["Churn"]

    print("\n" + "=" * 60)
    print("FEATURES AND TARGET")
    print("=" * 60)

    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples : {X.shape[0]}")

    return X, y


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

def identify_features(X):
    """
    Identify numerical and categorical columns.
    """

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print("\n" + "=" * 60)
    print("FEATURE TYPES")
    print("=" * 60)

    print("\nNumerical features:")

    for feature in numerical_features:
        print(f"  - {feature}")

    print("\nCategorical features:")

    for feature in categorical_features:
        print(f"  - {feature}")

    return numerical_features, categorical_features


# ============================================================
# CREATE PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor(
    numerical_features,
    categorical_features
):
    """
    Create preprocessing pipelines for numerical
    and categorical features.
    """

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Combine numerical and categorical pipelines
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess_data():
    """
    Complete preprocessing workflow.
    """

    # --------------------------------------------------------
    # Create models directory if it doesn't exist
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    df = clean_data(df)

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    X, y = split_features_target(df)

    # --------------------------------------------------------
    # Identify feature types
    # --------------------------------------------------------

    numerical_features, categorical_features = (
        identify_features(X)
    )

    # --------------------------------------------------------
    # Create preprocessing pipeline
    # --------------------------------------------------------

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)

    print(
        f"Training samples: {X_train.shape[0]}"
    )

    print(
        f"Testing samples : {X_test.shape[0]}"
    )

    # --------------------------------------------------------
    # Fit preprocessing ONLY on training data
    # --------------------------------------------------------

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Transform test data using the fitted preprocessor
    X_test_processed = preprocessor.transform(
        X_test
    )

    # --------------------------------------------------------
    # Save preprocessing pipeline
    # --------------------------------------------------------

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Processed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed testing shape: "
        f"{X_test_processed.shape}"
    )

    print(
        f"\nPreprocessor saved to:"
    )

    print(PREPROCESSOR_PATH)

    # --------------------------------------------------------
    # Return processed data
    # --------------------------------------------------------

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    preprocess_data()