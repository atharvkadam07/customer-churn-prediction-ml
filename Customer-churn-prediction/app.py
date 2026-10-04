"""
Customer Churn Prediction
Streamlit Web Application
"""

import os
import sys

import streamlit as st
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.join(
    CURRENT_DIR,
    "src"
)

sys.path.append(SRC_DIR)

from predict import predict_churn


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Main content */
    .main {
        color: #ffffff;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #a7adb8;
        margin-bottom: 30px;
    }

    /* Cards */
    .info-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
    }

    .result-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
    }

    .probability {
        font-size: 52px;
        font-weight: 700;
        margin: 10px 0;
    }

    .risk-high {
        font-size: 22px;
        font-weight: 700;
    }

    .risk-medium {
        font-size: 22px;
        font-weight: 700;
    }

    .risk-low {
        font-size: 22px;
        font-weight: 700;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
        font-size: 17px;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b0f14;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Machine Learning powered customer churn risk prediction
    using XGBoost.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the Model")

    st.write(
        """
        This application predicts the probability that
        a customer may churn based on their demographic,
        service, contract, and billing information.
        """
    )

    st.divider()

    st.subheader("Model")

    st.write("**Algorithm:** XGBoost")

    st.write("**Training samples:** 5,634")

    st.write("**Features after preprocessing:** 45")

    st.divider()

    st.subheader("Risk Levels")

    st.write("🔴 **High:** ≥ 70%")

    st.write("🟡 **Medium:** 40–69%")

    st.write("🟢 **Low:** < 40%")


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.markdown(
    "## 👤 Customer Information"
)

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Column 1
# ------------------------------------------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [
            0,
            1
        ],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        [
            "Yes",
            "No"
        ]
    )

    dependents = st.selectbox(
        "Dependents",
        [
            "Yes",
            "No"
        ]
    )

    tenure = st.slider(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )


# ------------------------------------------------------------
# Column 2
# ------------------------------------------------------------

with col2:

    phone_service = st.selectbox(
        "Phone Service",
        [
            "Yes",
            "No"
        ]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


# ------------------------------------------------------------
# Column 3
# ------------------------------------------------------------

with col3:

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# ============================================================
# BILLING INFORMATION
# ============================================================

st.markdown(
    "## 💳 Billing Information"
)

bill_col1, bill_col2 = st.columns(2)


with bill_col1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )


with bill_col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=10.0
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create customer dictionary
    # --------------------------------------------------------

    customer_data = {

        "gender": gender,

        "SeniorCitizen": senior_citizen,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone_service,

        "MultipleLines": multiple_lines,

        "InternetService": internet_service,

        "OnlineSecurity": online_security,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_protection,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless_billing,

        "PaymentMethod": payment_method,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges
    }

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    try:

        result = predict_churn(
            customer_data
        )

        probability = result[
            "probability_percentage"
        ]

        prediction = result[
            "prediction"
        ]

        risk_level = result[
            "risk_level"
        ]

        # ----------------------------------------------------
        # Result display
        # ----------------------------------------------------

        st.markdown(
            "## 🎯 Prediction Result"
        )

        result_col1, result_col2 = st.columns(2)

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        with result_col1:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### Churn Probability"
            )

            st.markdown(
                f'<div class="probability">'
                f'{probability:.2f}%'
                f'</div>',
                unsafe_allow_html=True
            )

            st.progress(
                int(probability)
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        with result_col2:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### Prediction"
            )

            if prediction == "Yes":

                st.error(
                    "⚠️ Customer is likely to churn"
                )

            else:

                st.success(
                    "✅ Customer is likely to stay"
                )

            st.markdown(
                f"### Risk Level: **{risk_level}**"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # Customer summary
        # ----------------------------------------------------

        st.markdown(
            "## 📋 Customer Summary"
        )

        summary_data = {
            "Feature": [
                "Tenure",
                "Contract",
                "Internet Service",
                "Monthly Charges",
                "Total Charges",
                "Payment Method"
            ],

            "Value": [
                f"{tenure} months",
                contract,
                internet_service,
                f"${monthly_charges:.2f}",
                f"${total_charges:.2f}",
                payment_method
            ]
        }

        summary_df = pd.DataFrame(
            summary_data
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction • "
    "Python • Pandas • Scikit-learn • XGBoost • Streamlit"
)