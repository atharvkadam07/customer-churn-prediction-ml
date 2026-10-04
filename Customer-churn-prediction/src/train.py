"""
Customer Churn Prediction
Model Training Pipeline

This module:
1. Loads and preprocesses the dataset
2. Handles class imbalance using SMOTE
3. Trains multiple classification models
4. Evaluates each model
5. Compares model performance
6. Selects XGBoost as the final model
7. Saves the trained model
8. Saves model comparison results
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


# ============================================================
# PROJECT PATH CONFIGURATION
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

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "churn_model.pkl"
)

RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)


# ============================================================
# IMPORT PREPROCESSING MODULE
# ============================================================

sys.path.append(CURRENT_DIR)

from preprocessing import preprocess_data


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    model_name,
    X_test,
    y_test
):
    """
    Evaluate a trained classification model.
    """

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Probability predictions
    # --------------------------------------------------------

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Churn",
                "Churn"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_models():

    print("\n")
    print("=" * 60)
    print("CUSTOMER CHURN MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Preprocess data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data()

    # --------------------------------------------------------
    # Step 2: Display original class distribution
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ORIGINAL TRAINING CLASS DISTRIBUTION")
    print("=" * 60)

    print(
        y_train.value_counts()
    )

    # --------------------------------------------------------
    # Step 3: Handle class imbalance using SMOTE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("APPLYING SMOTE")
    print("=" * 60)

    smote = SMOTE(
        random_state=42
    )

    X_train_balanced, y_train_balanced = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    print("\nClass distribution after SMOTE:")

    print(
        pd.Series(
            y_train_balanced
        ).value_counts()
    )

    print(
        f"\nOriginal training samples: "
        f"{len(y_train)}"
    )

    print(
        f"Balanced training samples: "
        f"{len(y_train_balanced)}"
    )

    # --------------------------------------------------------
    # Step 4: Create models
    # --------------------------------------------------------

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    }

    # --------------------------------------------------------
    # Step 5: Train and evaluate models
    # --------------------------------------------------------

    results = []

    trained_models = {}

    for model_name, model in models.items():

        print("\n")
        print("=" * 60)
        print(f"TRAINING: {model_name}")
        print("=" * 60)

        # Train
        model.fit(
            X_train_balanced,
            y_train_balanced
        )

        # Store trained model
        trained_models[model_name] = model

        # Evaluate
        metrics = evaluate_model(
            model,
            model_name,
            X_test,
            y_test
        )

        results.append(
            metrics
        )

    # --------------------------------------------------------
    # Step 6: Create comparison dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Step 7: Save comparison results
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        f"\nModel comparison saved to:"
    )

    print(
        RESULTS_PATH
    )

    # --------------------------------------------------------
    # Step 8: Select XGBoost as final model
    # --------------------------------------------------------

    final_model = trained_models[
        "XGBoost"
    ]

    # --------------------------------------------------------
    # Step 9: Save final model
    # --------------------------------------------------------

    joblib.dump(
        final_model,
        FINAL_MODEL_PATH
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL SAVED")
    print("=" * 60)

    print(
        "Final model: XGBoost"
    )

    print(
        f"Saved to: {FINAL_MODEL_PATH}"
    )

    print("\nTraining completed successfully!")

    return (
        results_df,
        final_model
    )


# ============================================================
# RUN TRAINING
# ============================================================

if __name__ == "__main__":

    train_models()