import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

st.title("AI Loyiha: Classification va Regression")

# CSV fayl yuklash
uploaded_file = st.file_uploader("CSV faylni yuklang", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.write("Yuklangan data:")
    st.dataframe(df)

    if st.button("Data preprocessing", key="preprocess_btn"):
        # Nan qiymatlarni tozalash
        df_cleaned = df.dropna()
        st.write("Tozalangan data:")
        st.dataframe(df_cleaned)

        # Avtomatik X va Y ajratish: Y = oxirgi ustun, X = qolganlari
        X = df_cleaned.iloc[:, :-1]
        y = df_cleaned.iloc[:, -1]

        # Agar label matn bo‘lsa, kodlash
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        st.session_state["X"] = X
        st.session_state["y"] = y
        st.success("X va Y avtomatik ajratildi! Endi modelni tanlang.")

    # Classification va Regression tanlash
    model_type = st.radio("Model turi:", ["Classification", "Regression"], key="model_type")

    # Modellar ro'yxati
    models = {
        "Classification": {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC()
        },
        "Regression": {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "SVR": SVR()
        }
    }

    # Model tanlash
    if "X" in st.session_state:
        model_name = st.selectbox("Modelni tanlang:", list(models[model_type].keys()), key="model_select")

        if st.button("Modelni o‘rgatish", key="train_btn"):
            model = models[model_type][model_name]
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state["X"], st.session_state["y"], test_size=0.2, random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if model_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                report = classification_report(y_test, y_pred)

                st.subheader("Model Baholash Natijalari (Classification)")
                st.write(f"Accuracy: {accuracy:.4f}")
                st.write(f"Precision: {precision:.4f}")
                st.write(f"Recall: {recall:.4f}")
                st.write(f"F1 Score: {f1:.4f}")

            else:
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                st.subheader("Model Baholash Natijalari (Regression)")
                st.write(f"MAE: {mae:.4f}")
                st.write(f"MSE: {mse:.4f}")
                st.write(f"RMSE: {rmse:.4f}")
                st.write(f"R² Score: {r2:.4f}")