import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BankRetain AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")


model = load_model()


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    section[data-testid="stSidebar"] {
        background-color: #11151c;
        border-right: 1px solid #252b35;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f5f7fa;
    }

    h1 {
        font-size: 2.8rem !important;
        font-weight: 750 !important;
        color: #f5f7fa !important;
    }

    h2 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #f5f7fa !important;
        margin-top: 2rem !important;
    }

    h3 {
        color: #f5f7fa !important;
    }

    div[data-testid="stMetric"] {
        background-color: #151a22;
        border: 1px solid #29313d;
        border-radius: 14px;
        padding: 22px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9aa7b8;
    }

    div[data-testid="stMetricValue"] {
        color: #f5f7fa;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 48px;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #29313d;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR - CUSTOMER INFORMATION
# ============================================================

with st.sidebar:

    st.markdown("## 👤 Customer Profile")

    st.caption(
        "Enter customer details to assess their banking churn risk."
    )

    st.divider()

    st.markdown("### Personal Information")

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    tenure = st.number_input(
        "Tenure (years)",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )

    st.markdown("### Banking Information")

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        max_value=300000.0,
        value=50000.0,
        step=1000.0
    )

    num_products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

    has_credit_card = st.selectbox(
        "Credit Card",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    active_member = st.selectbox(
        "Active Member",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        max_value=300000.0,
        value=75000.0,
        step=1000.0
    )

    st.divider()

    analyze_button = st.button(
        "🔍 Analyze Customer",
        type="primary",
        width="stretch"
    )


# ============================================================
# HEADER
# ============================================================

st.title("🏦 BankRetain AI")

st.write(
    "AI-powered banking customer retention analysis "
    "to identify customers who may be at risk of leaving."
)

st.divider()


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_credit_card],
    "IsActiveMember": [active_member],
    "EstimatedSalary": [estimated_salary]
})


# ============================================================
# INITIAL STATE
# ============================================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False


# ============================================================
# PREDICTION
# ============================================================

if analyze_button:

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        churn_percentage = probability * 100

        st.session_state.prediction_done = True
        st.session_state.prediction = prediction
        st.session_state.probability = churn_percentage
        st.session_state.customer_data = input_data.copy()

    except Exception as e:

        st.error(
            "Unable to generate the prediction. "
            "Please check that the model and input columns match."
        )

        st.exception(e)


# ============================================================
# SHOW DASHBOARD
# ============================================================

if st.session_state.prediction_done:

    prediction = st.session_state.prediction
    churn_percentage = st.session_state.probability
    customer_data = st.session_state.customer_data


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.header("🎯 Customer Risk Assessment")


    # --------------------------------------------------------
    # DETERMINE RISK LEVEL
    # --------------------------------------------------------

    if churn_percentage >= 70:

        risk = "High Risk"
        risk_icon = "🔴"

    elif churn_percentage >= 40:

        risk = "Medium Risk"
        risk_icon = "🟡"

    else:

        risk = "Low Risk"
        risk_icon = "🟢"


    # --------------------------------------------------------
    # PREDICTION TEXT
    # --------------------------------------------------------

    if prediction == 1:

        prediction_text = "⚠️ Likely to Churn"

    else:

        prediction_text = "✅ Likely to Stay"


    # ========================================================
    # THREE MAIN RESULT CARDS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🎯 Churn Probability",
            f"{churn_percentage:.2f}%"
        )


    with col2:

        st.metric(
            "⚠️ Prediction",
            prediction_text
        )


    with col3:

        st.metric(
            "Risk Level",
            f"{risk_icon} {risk}"
        )


    # ========================================================
    # RISK ANALYSIS
    # ========================================================

    st.header("📊 Risk Analysis")

    st.progress(
        min(int(churn_percentage), 100)
    )

    st.caption(
        f"Estimated probability of customer churn: "
        f"{churn_percentage:.2f}%"
    )


    # --------------------------------------------------------
    # RISK EXPLANATION
    # --------------------------------------------------------

    if churn_percentage >= 70:

        st.error(
            "🔴 **High Risk:** This customer has a high probability "
            "of leaving the bank. Immediate retention attention "
            "is recommended."
        )

    elif churn_percentage >= 40:

        st.warning(
            "🟡 **Medium Risk:** This customer shows some signs "
            "of potential churn. Proactive engagement may help "
            "retain the customer."
        )

    else:

        st.success(
            "🟢 **Low Risk:** This customer currently shows a "
            "relatively low probability of churn."
        )


    # ========================================================
    # RECOMMENDED ACTION
    # ========================================================

    st.header("💡 Recommended Action")


    if churn_percentage >= 70:

        st.error(
            "### Immediate Retention Action\n\n"
            "• Contact the customer proactively\n\n"
            "• Offer a personalized retention benefit\n\n"
            "• Review account activity and customer concerns\n\n"
            "• Consider a dedicated relationship-manager follow-up"
        )

    elif churn_percentage >= 40:

        st.warning(
            "### Proactive Engagement\n\n"
            "• Monitor customer activity\n\n"
            "• Offer relevant banking products or services\n\n"
            "• Send personalized communication\n\n"
            "• Follow up if customer engagement decreases"
        )

    else:

        st.success(
            "### Continue Normal Engagement\n\n"
            "• Maintain regular customer engagement\n\n"
            "• Continue providing relevant banking services\n\n"
            "• Monitor for future changes in customer behavior"
        )


    # ========================================================
    # CUSTOMER INFORMATION
    # ========================================================

    st.header("👤 Customer Information")

    display_data = customer_data.copy()

    display_data["HasCrCard"] = display_data["HasCrCard"].map(
        {0: "No", 1: "Yes"}
    )

    display_data["IsActiveMember"] = display_data[
        "IsActiveMember"
    ].map(
        {0: "No", 1: "Yes"}
    )

    display_data["Balance"] = display_data["Balance"].map(
        lambda x: f"${x:,.2f}"
    )

    display_data["EstimatedSalary"] = display_data[
        "EstimatedSalary"
    ].map(
        lambda x: f"${x:,.2f}"
    )

    st.dataframe(
        display_data,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # CUSTOMER PROFILE SUMMARY
    # ========================================================

    st.header("📋 Customer Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Credit Score",
            f"{credit_score}"
        )

    with col2:

        st.metric(
            "Age",
            f"{age} years"
        )

    with col3:

        st.metric(
            "Tenure",
            f"{tenure} years"
        )

    with col4:

        st.metric(
            "Products",
            f"{num_products}"
        )


    # ========================================================
    # ABOUT BANKRETAIN AI
    # ========================================================

    st.divider()

    with st.expander("ℹ️ About BankRetain AI"):

        st.subheader("🏦 BankRetain AI")

        st.write(
            "BankRetain AI is a machine learning based banking "
            "customer retention application. It estimates the "
            "probability that a customer may leave the bank based "
            "on their personal and banking information."
        )

        st.write("### 🤖 Machine Learning Model")

        st.write(
            "The application uses a trained Gradient Boosting "
            "classification model to generate customer churn "
            "predictions."
        )

        st.write("### 📊 Model Details")

        about_col1, about_col2, about_col3 = st.columns(3)

        with about_col1:

            st.metric(
                "🤖 Model",
                "Gradient Boosting"
            )

        with about_col2:

            st.metric(
                "🎯 Accuracy",
                "87.05%"
            )

        with about_col3:

            st.metric(
                "📈 ROC-AUC",
                "86.97%"
            )

        st.write(
            "These performance metrics are included here for "
            "machine learning project and academic presentation "
            "purposes. They are not displayed in the main "
            "customer-facing prediction dashboard."
        )

        st.write("### 🗂️ Dataset")

        st.write(
            "The model was trained using a bank customer dataset "
            "containing customer demographic, financial and "
            "account activity information."
        )

        st.write(
            "**Prediction Target:** Customer Churn (Exited)"
        )


# ============================================================
# INITIAL / EMPTY STATE
# ============================================================

else:

    st.header("🎯 Ready for Analysis")

    st.info(
        "Enter the customer's information in the panel on the "
        "left and click **Analyze Customer** to generate a "
        "banking customer churn prediction."
    )

    st.markdown("### What you will get")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🎯 Customer Risk Assessment")

        st.write(
            "A percentage showing how likely the customer "
            "is to leave the bank."
        )

    with col2:

        st.markdown("### ⚠️ Risk Level")

        st.write(
            "The customer will be classified as Low, Medium, "
            "or High Risk."
        )

    with col3:

        st.markdown("### 💡 Recommended Action")

        st.write(
            "The dashboard will provide a suitable customer "
            "retention recommendation."
        )


    # ========================================================
    # ABOUT PROJECT - BEFORE PREDICTION
    # ========================================================

    st.divider()

    with st.expander("ℹ️ About BankRetain AI"):

        st.subheader("🏦 BankRetain AI")

        st.write(
            "BankRetain AI is an AI-powered banking customer "
            "retention system designed to identify customers "
            "who may be at risk of leaving the bank."
        )

        st.write(
            "The system analyzes customer information such as "
            "credit score, age, geography, account balance, "
            "number of products, credit card status, active "
            "membership and estimated salary."
        )

        st.write("### 🤖 Machine Learning Model")

        st.write(
            "A trained Gradient Boosting classification model "
            "is used to estimate the probability of customer "
            "churn."
        )

        # ----------------------------------------------------
        # ML DETAILS FOR COLLEGE PRESENTATION
        # ----------------------------------------------------

        st.write("### 📊 Model Details")

        about_col1, about_col2, about_col3 = st.columns(3)

        with about_col1:

            st.metric(
                "🤖 Model",
                "Gradient Boosting"
            )

        with about_col2:

            st.metric(
                "🎯 Accuracy",
                "87.05%"
            )

        with about_col3:

            st.metric(
                "📈 ROC-AUC",
                "86.97%"
            )

        st.write(
            "These model performance metrics are kept inside "
            "the About section for academic demonstration "
            "and project presentation purposes."
        )

        st.write("### 🗂️ Dataset")

        st.write(
            "Bank customer information is used to train the "
            "machine learning model and predict customer churn."
        )

        st.write(
            "**Prediction Target:** Customer Churn (Exited)"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#7f8b9a;
        padding:30px 0 10px 0;
        font-size:0.85rem;
    ">
        BankRetain AI • Banking Customer Retention System<br>
        Built with Python, Scikit-learn & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)