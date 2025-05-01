import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ----------------------
# Load Model and Scaler
# ----------------------
model = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------
# Streamlit Page Config
# ----------------------
st.set_page_config(page_title="Retail Sales Prediction", layout="wide")

# ----------------------
# Sidebar - Navigation
# ----------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to:", ["📈 Prediction App", "🔍 About the Model", "📊 Business Impact", "🧠 SHAP Explainability"])

# ----------------------
# Section: Prediction App
# ----------------------
if app_mode == "📈 Prediction App":
    st.title("🛍️ Retail Sales Prediction with XGBoost")
    st.markdown("Upload your retail transaction data or use the default dataset to predict **Sales Amount** per transaction.")

    st.subheader("📂 Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose a file", type="csv")

    # Load dataset
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Custom dataset uploaded!")
    else:
        df = pd.read_csv("Retail_Sales.csv")
        st.info("⚠ Using default dataset: `Retail_Sales.csv`")

    # Preprocessing
    raw = df.copy()
    raw.drop(columns=["Unnamed: 0", "Date", "Transaction_ID", "SKU"], inplace=True, errors="ignore")

    for col in ["Customer_ID", "SKU_Category"]:
        raw[col] = pd.factorize(raw[col])[0]

    X = raw.drop(columns="Sales_Amount")
    y_true = raw["Sales_Amount"]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    df["Predicted_Sales"] = y_pred

    # Display Data
    st.subheader("📊 Predicted Sales Output")
    st.write(df[["Customer_ID", "SKU_Category", "Quantity", "Sales_Amount", "Predicted_Sales"]].head(10))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    st.metric("📉 Mean Absolute Error (MAE)", f"${mae:.2f}")

    st.markdown(f"""
    **MAE Interpretation**: On average, the model's predicted sales deviate from the actual by **${mae:.2f}** per transaction.
    - If avg sale = $25 → this model has approx **{round((mae/25)*100)}% error rate**.
    - Lower MAE means more reliable revenue forecasts.

    ✅ Why is MAE shown in dollars ($)?                                                                  
    The model predicts **Sales_Amount**, which is a **monetary value**, so MAE is also in **dollars**.

    🔎 **Example:**
    - True Sales: **$50**
    - Predicted: **$38**
    - Absolute Error: **|50 - 38| = $12**
    - Averaging all such errors gives the **MAE**.

    - **Random Forest MAE**: ~$9.21  
    - **XGBoost MAE**: ~$8.67 → more accurate  
    - **Current MAE**: **${mae:.2f}**

    ✅ It’s in **dollars** because the model is trained to predict monetary outcomes.
    """)

# ----------------------
# Section: SHAP Explainability
# ----------------------
elif app_mode == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Feature Importance")
    st.markdown("Visual explanation of what influences the model's predictions the most.")

    st.image("images/SHAP_value.png", caption="SHAP Summary Plot — What drives sales prediction?", use_container_width=True)

    st.markdown("""
    ### 🧾 Interpretation:
    - **Rolling sales trends** (e.g., 7-day average) and **recent lag values** are top influencers.
    - Time-based features (e.g., `day_of_week`, `month`, `holiday_flag`) also significantly affect predictions.
    - Understanding SHAP values helps non-technical stakeholders with and transparency and trust model decisions.
    """)

# ----------------------
# Section: About the Model
# ----------------------
elif app_mode == "🔍 About the Model":
    st.title("📌 About This Project")
    st.markdown("""
    - Model: This app uses **XGBoost Regression** to predict transaction-level sales.
    - Dataset: Trained on real-world retail sales data from Kaggle.
    - Metric: Evaluated using **Mean Absolute Error (MAE)** to measure average prediction error.
    - Explainability: Includes **SHAP explainability** for model transparency.
    """)

    st.subheader("🧪 Model Evaluation")
    st.markdown("""
    - ✅ **Random Forest MAE**: $9.21
    - ✅ **XGBoost MAE**: $8.67 → better performance & better prediction accuracy.
    - ✔️ XGBoost captures complex patterns with lower error, making it suitable for sales forecasting.
    """)

# ----------------------
# Section: Business Impact
# ----------------------
elif app_mode == "📊 Business Impact":
    st.title("💼 Business Insights & Recommendations")
    st.markdown("""
    ### 💰 Impact of Model Adoption:
    - Helps similar businesses like **Amazon**, **Netflix**, and e-commerce platforms optimize operations, inventory, and reduce over/understock.
    - Can be used to predict not only sales but also **customer behavior** and **content popularity**.
    - Enables **targeted promotions** by predicting sales patterns.
    - For businesses doing 50M dollars/month, a 5 percent lift equals **2.5M dollars additional revenue/month**.

    ### 🔁 Suitable For:
    - Retail Chains, E-commerce Platforms
    - Fulfillment Platforms
    - Subscription & Streaming Services (adapt prediction logic to content engagement & modeling)

    ### 🧠 Strategic Recommendations:
    - Use SHAP to identify **top-selling patterns** and interpret what's driving performance.                                      
    - Re-train periodically with new data to adapt to market shifts and consumer patterns.                                      
    - Embed in BI tools and Integrate with dashboards (Tableau, Power BI) for decision-making and C-suite visibility.   

    ### 🧾 Business Uplift:
    - Sales prediction = improved planning
    - MAE reduction = better accuracy and confidence
    - Estimated revenue gain: **$2.5M/month** for medium retailers                           

    ### 🧾 Recruiter-Ready Highlights:
    - Full ML pipeline: preprocessing → training → deployment
    - Business storytelling + measurable outcomes
    - Adaptable to multiple industries (retail, streaming, logistics)
    """)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("Developed by **Sweety Seelam** | MIT License | © 2025")