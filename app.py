# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# Load dataset and model
# -----------------------------
@st.cache_data
def load_data():
    file_path = "data/Titanic-Dataset.csv"
    if not os.path.exists(file_path):
        st.error("❌ Dataset not found! Please place 'Titanic-Dataset.csv' inside the 'data/' folder.")
        st.stop()
    df = pd.read_csv(file_path)
    return df

@st.cache_resource
def load_model():
    file_path = "model.pkl"
    if not os.path.exists(file_path):
        st.error("❌ Model file not found! Please place 'model.pkl' in the project root.")
        st.stop()
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Home", "Data Exploration", "Visualizations", "Model Prediction", "Model Performance"]
)

# -----------------------------
# Home
# -----------------------------
if menu == "Home":
    st.title("🚢 Titanic Survival Prediction App")
    st.markdown("""
    Welcome to the Titanic Survival Prediction App!  
    This app demonstrates a **machine learning model** deployed with **Streamlit**.  

    **Features:**
    - Explore Titanic dataset
    - Interactive visualizations
    - Make survival predictions
    - Check model performance
    """)

# -----------------------------
# Data Exploration
# -----------------------------
elif menu == "Data Exploration":
    st.title("🔎 Data Exploration")

    st.subheader("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Filter Data")
    pclass = st.multiselect("Select Passenger Class", df["Pclass"].unique(), default=df["Pclass"].unique())
    sex = st.multiselect("Select Sex", df["Sex"].unique(), default=df["Sex"].unique())
    filtered_df = df[(df["Pclass"].isin(pclass)) & (df["Sex"].isin(sex))]
    st.dataframe(filtered_df)

# -----------------------------
# Visualizations
# -----------------------------
elif menu == "Visualizations":
    st.title("📊 Visualizations")

    st.subheader("Survival Count")
    fig1 = px.histogram(df, x="Survived", color="Sex", barmode="group")
    st.plotly_chart(fig1)

    st.subheader("Class vs Survival")
    fig2 = px.histogram(df, x="Pclass", color="Survived", barmode="group")
    st.plotly_chart(fig2)

    st.subheader("Age Distribution")
    fig3 = px.histogram(df, x="Age", nbins=30, color="Survived")
    st.plotly_chart(fig3)

# -----------------------------
# Model Prediction
# -----------------------------
elif menu == "Model Prediction":
    st.title("🤖 Model Prediction")

    st.write("Enter passenger details to predict survival:")

    Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    Sex = st.selectbox("Sex", ["male", "female"])
    Age = st.slider("Age", 0, 80, 25)
    SibSp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
    Parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
    Fare = st.slider("Passenger Fare", 0, 500, 50)
    Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # Encode inputs same way as training
    sex_encoded = 1 if Sex == "male" else 0
    embarked_mapping = {"C": 0, "Q": 1, "S": 2}
    embarked_encoded = embarked_mapping[Embarked]

    features = np.array([[Pclass, sex_encoded, Age, SibSp, Parch, Fare, embarked_encoded]])

    if st.button("Predict Survival"):
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        if prediction == 1:
            st.success(f"✅ Passenger Survives! (Probability: {prob[1]:.2f})")
        else:
            st.error(f"❌ Passenger Does Not Survive (Probability: {prob[0]:.2f})")

# -----------------------------
# Model Performance
# -----------------------------
elif menu == "Model Performance":
    st.title("📈 Model Performance")

    # Prepare test data same way as training
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[features].copy()
    y = df["Survived"]

    X["Sex"] = LabelEncoder().fit_transform(X["Sex"])
    X["Embarked"] = LabelEncoder().fit_transform(X["Embarked"].astype(str))

    imputer = SimpleImputer(strategy="median")
    X["Age"] = imputer.fit_transform(X[["Age"]])
    X["Fare"] = imputer.fit_transform(X[["Fare"]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    st.subheader("Accuracy Score")
    st.write(accuracy_score(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
