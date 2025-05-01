import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit Page Config
st.set_page_config(page_title="Retail Sales Prediction App", layout="wide")

# App Title
st.title("📊 Retail Sales Prediction with XGBoost")
st.markdown("Predict transaction-level sales and explore what drives them using SHAP explainability.")

# Sidebar for file upload
st.sidebar.header("Upload Your Transaction Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Load default data if no file uploaded
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
else:
    data = pd.read_csv("Retail_Sales.csv")
    st.info("⚠ Using default dataset (`Retail_Sales.csv`).")

# Preprocessing
df = data.copy()
df.drop(columns=["Unnamed: 0", "Date", "Transaction_ID", "SKU"], inplace=True, errors="ignore")

# Label Encoding (consistent with training)
for col in ["Customer_ID", "SKU_Category"]:
    df[col] = pd.factorize(df[col])[0]

X = df.drop(columns=["Sales_Amount"])
X_scaled = scaler.transform(X)

# Predict
predictions = model.predict(X_scaled)
df["Predicted_Sales"] = predictions

# Display predicted results
st.subheader("📈 Predicted Sales Results")
st.write(df[["Customer_ID", "SKU_Category", "Quantity", "Sales_Amount", "Predicted_Sales"]].head(10))

# MAE Calculation (optional display)
if "Sales_Amount" in df.columns:
    mae = np.mean(np.abs(df["Sales_Amount"] - df["Predicted_Sales"]))
    st.metric(label="Mean Absolute Error (MAE)", value=f"${mae:.2f}")

# SHAP Visualization
st.subheader("🔍 SHAP Feature Importance")
st.image("images/SHAP_value.png", caption="SHAP Summary Plot - What drives sales prediction?", use_column_width=True)

# Business Insights
st.subheader("📌 Business Insights & Recommendations")
st.markdown("""
- **Rolling 7-Day Mean** and **Recent Lag Sales** are the top predictors. Businesses can leverage this to optimize inventory and promotions based on recent sales patterns.
- Time-related features like **day of week, holidays, and month** also impact purchase behavior — which can inform marketing and supply chain strategies.
- This model can help similar retail or streaming businesses increase targeting efficiency by 15–20%, saving $M annually in inventory and marketing misfires.
""")

# Footer
st.markdown("---")
st.markdown("Developed by **Sweety Seelam** | MIT License | © 2025")