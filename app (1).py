import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)

st.set_page_config(layout="wide")
st.title("🚀 ML Model Trainer Dashboard")

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ---------------- SIDEBAR CONTROLS ----------------

    st.sidebar.header("⚙️ Model Configuration")

    model_type = st.sidebar.radio(
        "Select Model Type",
        ["Classification", "Regression"]
    )

    # -------- Target Filtering --------
    if model_type == "Classification":
        possible_targets = [
            col for col in df.columns
            if df[col].dtype == "object"
            or (np.issubdtype(df[col].dtype, np.integer) and df[col].nunique() <= 10)
        ]
    else:
        possible_targets = [
            col for col in df.columns
            if np.issubdtype(df[col].dtype, np.number) and df[col].nunique() > 10
        ]

    if len(possible_targets) == 0:
        st.error("No suitable target columns found.")
        st.stop()

    target = st.sidebar.selectbox("Select Target Variable", possible_targets)

    features = st.sidebar.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target]
    )

    test_size = st.sidebar.slider(
        "Test Size (%)",
        10, 50, 20
    )

    train_button = st.sidebar.button("Train & Evaluate Model")

    # ---------------- MAIN AREA LAYOUT ----------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df)

    with col2:

        if train_button:

            if len(features) == 0:
                st.warning("Please select feature columns.")
                st.stop()

            X = df[features]
            y = df[target]

            X = pd.get_dummies(X, drop_first=True)

            # ---------------- CLASSIFICATION ----------------
            if model_type == "Classification":

                y_encoded, _ = pd.factorize(y)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded,
                    test_size=test_size / 100,
                    random_state=42
                )

                model = GaussianNB()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                st.subheader("📈 Classification Results")

                st.metric("Accuracy", f"{acc:.4f}")

                fig, ax = plt.subplots()
                ax.imshow(cm)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")

                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, cm[i, j],
                                ha="center", va="center")

                st.pyplot(fig)

                st.subheader("Classification Report")
                st.text(classification_report(y_test, y_pred))

            # ---------------- REGRESSION ----------------
            else:

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size / 100,
                    random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.subheader("📈 Regressxion Results")

                col_m1, col_m2 = st.columns(2)
                col_m1.metric("MSE", f"{mse:.4f}")
                col_m2.metric("R² Score", f"{r2:.4f}")

                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)

else:
    st.info("Upload a CSV file from the sidebar to begin.")