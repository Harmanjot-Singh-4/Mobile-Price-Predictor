import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------
# 1. Custom Transformers (Required for joblib to deserialize)
# ---------------------------------------------------------
class BrandValueAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.brand_value_map = {
            'Apple': 9.8, 'Samsung': 9.0, 'Google': 8.5, 'OnePlus': 8.0,
            'Sony': 7.5, 'Asus': 7.2, 'Xiaomi': 7.0, 'Huawei': 6.8,
            'Oppo': 6.5, 'Vivo': 6.5, 'Motorola': 6.2, 'Realme': 6.0,
            'Nokia': 5.5, 'Blackberry': 5.0, 'LG': 5.0
        }
        self.default_score = 5.0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out['Brand Value Score'] = (
            X_out['Brand'].astype(str).str.strip()
            .map(self.brand_value_map)
            .fillna(self.default_score)
        )
        return X_out

class ServiceScoreAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.service_score_map = {
            'Apple': 9.5, 'Samsung': 9.0, 'Xiaomi': 8.2, 'Oppo': 7.8,
            'Vivo': 7.8, 'OnePlus': 7.5, 'Realme': 7.5, 'Google': 7.0,
            'Motorola': 7.0, 'Asus': 6.8, 'Nokia': 6.5, 'Sony': 6.2,
            'Huawei': 6.0, 'LG': 5.0, 'Blackberry': 4.0
        }
        self.default_score = 5.0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out['Service Score'] = (
            X_out['Brand'].astype(str).str.strip()
            .map(self.service_score_map)
            .fillna(self.default_score)
        )
        return X_out

class SpecTierClassifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        def assign_tier(row):
            ram = row.get('RAM (GB)', 4)
            storage = row.get('Storage (GB)', 64)
            if ram >= 8 and storage >= 256:
                return 'Flagship'
            elif ram <= 4 and storage <= 64:
                return 'Budget'
            else:
                return 'Mid-Range'
                
        X_out['Spec Tier'] = X_out.apply(assign_tier, axis=1)
        return X_out

# ---------------------------------------------------------
# 2. Page Configuration & UI Layout
# ---------------------------------------------------------
st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="centered")

st.title("📱 Mobile Price Predictor")
st.write("Predict smartphone market prices using machine learning regression and custom brand-tier feature engineering.")
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
    
    # Locate mobile_price_pipeline.pkl anywhere in the repository structure
    target_name = "mobile_price_pipeline.pkl"
    model_file = None

    # Check root working directory and script directory
    # Fast, direct path lookup (no deep directory scanning)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "mobile_price_pipeline.pkl"),
        "mobile_price_pipeline.pkl",
        "Mobile Price Predictor/mobile_price_pipeline.pkl",
        os.path.join(script_dir, "..", "mobile_price_pipeline.pkl"),
    ]
    model_file = next((p for p in candidates if os.path.exists(p)), None)
    
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
