import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------
# 1. Custom Transformers (Required for joblib to deserialize)
# ---------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Custom Transformer: Brand Value
class BrandValueAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.brand_scores = {
            'Apple': 9.8, 'Samsung': 9.0, 'Google': 8.5, 
            'OnePlus': 8.0, 'Xiaomi': 7.2, 'Oppo': 6.8, 
            'Vivo': 6.8, 'Realme': 6.5, 'Motorola': 6.0
        }
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['Brand_Value'] = X_copy['Brand'].map(self.brand_scores).fillna(5.0)
        return X_copy

# 2. Custom Transformer: Service Score
class ServiceScoreAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.service_scores = {
            'Apple': 9.5, 'Samsung': 9.0, 'OnePlus': 8.0, 
            'Xiaomi': 8.2, 'Oppo': 7.5, 'Vivo': 7.5, 
            'Google': 7.0, 'Realme': 7.0, 'Motorola': 6.5
        }
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['Service_Score'] = X_copy['Brand'].map(self.service_scores).fillna(4.0)
        return X_copy

# 3. Custom Transformer: Spec Tier
class SpecTierClassifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        def classify_tier(row):
            ram = row['RAM (GB)']
            storage = row['Storage (GB)']
            if ram >= 8 and storage >= 256:
                return 'Flagship'
            elif ram >= 6 or storage >= 128:
                return 'Mid-Range'
            else:
                return 'Budget'
        X_copy['Spec_Tier'] = X_copy.apply(classify_tier, axis=1)
        return X_copy

# ---------------------------------------------------------
# 2. Page Configuration & UI Layout
# ---------------------------------------------------------
st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="centered")

st.title("📱 Real-World Mobile Price Predictor")
st.write("Predict smartphone market prices using brand tiering and machine learning regression.")
st.divider()

col1, col2 = st.columns(2)

brand_options = [
    "Apple", "Samsung", "Google", "OnePlus", "Sony", 
    "Xiaomi", "Asus", "Oppo", "Vivo", "Realme", 
    "Motorola", "Nokia", "Huawei", "Blackberry", "LG"
]

with col1:
    brand = st.selectbox("Brand", sorted(brand_options), index=0)
    ram = st.selectbox("RAM (GB)", [2, 3, 4, 6, 8, 12, 16], index=3)
    storage = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512], index=2)

with col2:
    screen_size = st.slider("Screen Size (inches)", min_value=4.5, max_value=7.6, value=6.5, step=0.1)
    battery = st.slider("Battery Capacity (mAh)", min_value=1800, max_value=7000, value=4500, step=100)
    cameras = st.selectbox("Number of Rear Cameras", [1, 2, 3, 4], index=2)

st.divider()

# ---------------------------------------------------------
# 3. Model Inference (Direct Disk Loading)
# ---------------------------------------------------------
if st.button("Predict Price", type="primary", use_container_width=True):
    raw_df = pd.DataFrame([{
        "Brand": brand,
        "Storage (GB)": storage,
        "RAM (GB)": ram,
        "Screen Size (inches)": screen_size,
        "Battery Capacity (mAh)": battery,
        "Number of Rear Cameras": cameras
    }])
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(base_dir, "mobile_price_pipeline.pkl")

    if os.path.exists(model_file):
        loaded_model = joblib.load(model_file)
        prediction = loaded_model.predict(raw_df)
        predicted_price = float(prediction[0])
    else:
        st.error("Model file 'mobile_price_pipeline.pkl' not found.")
        predicted_price = 450.0
    
    # 95% Confidence Interval based on test MAE (~$59.39)
    margin = 1.96 * 59.39
    min_price = max(50.0, predicted_price - margin)
    max_price = predicted_price + margin
    
    if predicted_price < 250:
        category = "Budget / Entry-Level 🟢"
    elif predicted_price < 650:
        category = "Mid-Range Value 🟡"
    else:
        category = "Premium / Flagship 🔴"

    st.subheader("Price Estimate")
    st.metric(label="Estimated Market Price", value=f"${predicted_price:,.2f}")
    
    st.info(
        f"**Category:** {category}\n\n"
        f"**Expected Range (95% Confidence):** ${min_price:,.2f} – ${max_price:,.2f}"
    )
