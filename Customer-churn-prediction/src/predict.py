"""
Customer Churn Prediction
Prediction Module

This module:
1. Loads the trained preprocessing pipeline
2. Loads the trained XGBoost model
3. Accepts customer information
4. Preprocesses the input
5. Predicts churn
6. Returns churn probability
"""

import os
import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "churn_model.pkl"
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    return model


def load_preprocessor():

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found at: "
            f"{PREPROCESSOR_PATH}"
        )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return preprocessor


# ============================================================
# PREPARE CUSTOMER INPUT
# ============================================================

def prepare_customer_data(customer_data):
    """
    Convert customer information into a DataFrame
    with the same structure used during training.
    """

    data = pd.DataFrame(
        [customer_data]
    )

    return data


# ============================================================
# PREDICT CHURN
# ============================================================

def predict_churn(customer_data):
    """
    Predict whether a customer is likely to churn.

    Returns:
        prediction
        probability
        risk_level
    """

    # --------------------------------------------------------
    # Load model and preprocessor
    # --------------------------------------------------------

    model = load_model()

    preprocessor = load_preprocessor()

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    data = prepare_customer_data(
        customer_data
    )

    # --------------------------------------------------------
    # Apply preprocessing
    # --------------------------------------------------------

    processed_data = preprocessor.transform(
        data
    )

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(
        processed_data
    )[0]

    # --------------------------------------------------------
    # Get churn probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        processed_data
    )[0][1]

    probability_percentage = (
        probability * 100
    )

    # --------------------------------------------------------
    # Determine risk level
    # --------------------------------------------------------

    if probability_percentage >= 70:

        risk_level = "High"

    elif probability_percentage >= 40:

        risk_level = "Medium"

    else:

        risk_level = "Low"

    # --------------------------------------------------------
    # Convert prediction to label
    # --------------------------------------------------------

    prediction_label = (
        "Yes"
        if prediction == 1
        else "No"
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "prediction": prediction_label,
        "probability": probability,
        "probability_percentage": probability_percentage,
        "risk_level": risk_level
    }


# ============================================================
# TEST PREDICTION
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example customer
    # --------------------------------------------------------

    example_customer = {

        "gender": "Female",

        "SeniorCitizen": 0,

        "Partner": "Yes",

        "Dependents": "No",

        "tenure": 5,

        "PhoneService": "Yes",

        "MultipleLines": "No",

        "InternetService": "Fiber optic",

        "OnlineSecurity": "No",

        "OnlineBackup": "No",

        "DeviceProtection": "No",

        "TechSupport": "No",

        "StreamingTV": "Yes",

        "StreamingMovies": "Yes",

        "Contract": "Month-to-month",

        "PaperlessBilling": "Yes",

        "PaymentMethod": "Electronic check",

        "MonthlyCharges": 85.50,

        "TotalCharges": 430.00
    }

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    result = predict_churn(
        example_customer
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 60)

    print(
        f"Prediction        : "
        f"{result['prediction']}"
    )

    print(
        f"Churn Probability : "
        f"{result['probability_percentage']:.2f}%"
    )

    print(
        f"Risk Level        : "
        f"{result['risk_level']}"
    )

    print("=" * 60)