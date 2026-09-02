# Mobile-Price-Predictor
End-to-end smartphone price prediction pipeline featuring custom feature engineering, log-transformed Ridge regression (~92% R²), and an interactive Streamlit web app.


Mobile Price Predictor: End-to-End Machine Learning Pipeline & Web App
An end-to-end regression pipeline designed to predict smartphone market prices based on hardware specifications, brand equity, and commercial tiering. The project encompasses rigorous data cleaning, deterministic hash-based data splitting, custom feature engineering, target log-transformation, and a mathematical verification using Batch Gradient Descent built from scratch. The final model achieves an R 
2
  score of 92.23% with an average error of ~$59, serialized and served via an interactive Streamlit web application.  
IPYNB
+ 2

Table of Contents
Project Overview

1. Data Cleaning & Preprocessing

2. Deterministic Train/Test Splitting (CRC32 Hash)

3. Custom Feature Engineering & Transformers

4. Model Selection & Cross-Validation

5. Target Log-Transformation & Gradient Descent from Scratch

6. Final Model Performance & Scorecard

7. Production Serialization & Streamlit Deployment

8. Local Installation & Setup

9. Future Roadmap & Enhancements

Project Overview
Smartphone pricing is influenced by both raw hardware performance and intangible market factors like brand perception and product tiering. Standard linear models often underperform on raw price targets due to extreme right-skewness caused by high-end flagships. This project addresses these challenges through a modular Scikit-Learn pipeline that handles feature extraction, market scoring, target transformation, and interactive real-time inference.  
IPYNB
+ 2

Raw CSV Dataset 
   │
   ▼
Regex Parsing & Deduplication (358 clean rows)
   │
   ▼
Deterministic CRC32 Hash Split (300 Train / 58 Test)
   │
   ▼
Custom Scikit-Learn Transformers (Brand Value, Service Score, Spec Tier)
   │
   ▼
ColumnTransformer (StandardScaler + OneHotEncoder -> 25 Features)
   │
   ▼
TransformedTargetRegressor (Ridge Regression + log1p / expm1)
   │
   ▼
Validation (NumPy Batch Gradient Descent from Scratch)
   │
   ▼
Serialization (joblib) ──▶ Streamlit Web App (app.py)
1. Data Cleaning & Preprocessing
The raw dataset contained 407 entries across 8 attributes, containing embedded units, inconsistent column labels, and multi-camera strings:  
IPYNB

Label Normalization: Cleaned inconsistent whitespaces, standardizing "Storage " to "Storage (GB)" and "RAM " to "RAM (GB)".  
IPYNB

Regex Feature Extraction: Extracted integer values from storage and RAM strings, while converting Screen Size (inches) to standard numeric floats.  
IPYNB

Target Parsing (Price ($)): Stripped currency symbols ($), commas, and whitespaces before casting prices to 64-bit integers.  
IPYNB

Camera Count Engineering: Parsed raw strings like "108 + 10 + 10 + 12" by counting + delimiters to engineer a discrete Number of Rear Cameras feature.  
IPYNB

Deduplication: Identified and purged 49 duplicate device records, yielding a clean dataset of 358 unique smartphones.  
IPYNB

2. Deterministic Train/Test Splitting (CRC32 Hash)
To prevent data leakage and guarantee reproducibility without relying on arbitrary random seeds, splitting was implemented using an identifier hash algorithm:  
IPYNB

Unique Device Hash: Generated a composite key per phone: Brand + Model + Storage (GB) + RAM (GB).  
IPYNB

CRC32 Partitioning: Hashed each identifier using a 32-bit Cyclic Redundancy Check (crc32). Devices with hash values below 20% of 2 
32
  were routed to the test set.  
IPYNB
+ 1

Data Allocation: Yielded a balanced, immutable split of 300 training samples and 58 hold-out test samples.  
IPYNB

3. Custom Feature Engineering & Transformers
Three custom Scikit-Learn transformers (BaseEstimator, TransformerMixin) were engineered to inject domain knowledge into the modeling process:  
IPYNB

BrandValueAdder: Quantifies brand equity and consumer willingness-to-pay (e.g., Apple: 9.8, Samsung: 9.0, Google: 8.5, OnePlus: 8.0 down to budget tiers at 5.0).  
IPYNB

ServiceScoreAdder: Maps after-sales support networks and warranty reliability (e.g., Apple: 9.5, Xiaomi: 8.2, down to 4.0).  
IPYNB

SpecTierClassifier: Categorizes devices into commercial tiers (Flagship, Mid-Range, Budget) using RAM and internal storage decision boundaries.  
IPYNB

These transformers feed directly into a unified ColumnTransformer that standardizes continuous variables with StandardScaler and expands categorical columns via OneHotEncoder, producing a feature matrix of 25 columns.  
IPYNB

4. Model Selection & Cross-Validation
A 5-fold cross-validation benchmark was performed on the training set to compare baseline algorithms:  
IPYNB

Algorithm	Train RMSE	5-Fold CV Mean RMSE	CV Standard Deviation	Generalization Behavior
Decision Tree Regressor	
$20.99  
IPYNB

$138.76  
IPYNB

$74.16  
IPYNB

Heavy memorization; high variance across folds.  
IPYNB

Linear Regression (Baseline)	
$117.51  
IPYNB

$150.66  
IPYNB

$50.97  
IPYNB

Stable fit, but hindered by limited feature inputs.  
IPYNB

Linear Regression + Engineered Features	
$108.21

  
IPYNB

$140.21

  
IPYNB

$62.15

  
IPYNB

Custom domain features reduced CV error by $10.45.  
IPYNB

5. Target Log-Transformation & Gradient Descent from Scratch
Smartphone retail prices are right-skewed, ranging from $99 to $1,999. Ordinary linear regression over-penalizes absolute dollar mistakes on expensive phones while under-optimizing budget tiers.  
IPYNB
+ 1

TransformedTargetRegressor
To optimize relative percentage error, the target variable was wrapped using logarithmic scaling:

Python
base_ridge = Ridge(alpha=0.5, random_state=42)
log_model = TransformedTargetRegressor(
    regressor=base_ridge, 
    func=np.log1p, 
    inverse_func=np.expm1
)
Log Compression: Fits the model on z=ln(1+price) during training.  
IPYNB

Automatic Inversion: Converts log predictions back into real-world dollar values (e 
z
 −1) during inference.  
IPYNB

Validation via NumPy Batch Gradient Descent
To verify Scikit-Learn's closed-form solution, a Batch Gradient Descent engine was implemented from scratch:  
IPYNB

Built an (m×26) matrix by prepending a bias vector (x 
0
​
 =1).  
IPYNB

Updated parameters over 2,000 epochs (η=0.05) using the analytical matrix derivative:
  
IPYNB

∇ 
θ
​
 MSE= 
m
2
​
 X 
b
T
​
 (X 
b
​
 θ−y 
log
​
 )
Verification: The manual NumPy implementation settled at an intercept of 4.1368 (baseline price ~$61.60) and matched the production library metrics within cents (92.31% R 
2
 ).  
IPYNB

6. Final Model Performance & Scorecard
Evaluating the models on the 58 hold-out test samples confirmed generalization on unseen data:  
IPYNB

Metric	Untransformed Linear Regression	Log-Transformed Ridge	Scratch Gradient Descent
Variance Explained (R 
2
 )	
85.94%  
IPYNB

92.23%

  
IPYNB

92.31%

  
IPYNB

Root Mean Squared Error (RMSE)	
$125.99  
IPYNB

$93.63

  
IPYNB

$93.14

  
IPYNB

Mean Absolute Error (MAE)	
$74.41  
IPYNB

$59.39

  
IPYNB

$59.07

  
IPYNB

Predictions within ±10% of Actual	
32.8%  
IPYNB

41.4%

  
IPYNB

—
Predictions within ±15% of Actual	
46.6%  
IPYNB

51.7%

  
IPYNB

—
Relative Prediction Accuracy (1−MAPE)	
78.72%  
IPYNB

84.07%

  
IPYNB

84.04%

  
IPYNB

Accuracy: Captures over 92% of market price variance across competitive tiers.  
IPYNB

Real-World Precision: Average price estimation error dropped by $15.02 down to $59.39.  
IPYNB

Reliability: Over half of all unseen test predictions land within ±15% of true retail price.  
IPYNB

7. Production Serialization & Streamlit Deployment
Unified Artifact: The full preprocessing pipeline and regularized regressor were serialized into mobile_price_pipeline.pkl using joblib.  
IPYNB

Interactive UI (app.py): Built with Streamlit to accept raw user specifications (brand, RAM, storage, battery, display size, and camera count).  
IPYNB

Confidence Interval: Displays point estimates alongside a 95% confidence interval calculated from test MAE (±1.96×$59.39).  
IPYNB

Dynamic Categorization: Assigns market segment tags (Budget / Entry-Level 🟢, Mid-Range Value 🟡, Premium / Flagship 🔴) based on predicted valuations.  
IPYNB

8. Local Installation & Setup
Prerequisites
Ensure Python 3.9+ is installed on your machine.

1. Clone the Repository
Bash
git clone https://github.com/<YOUR_USERNAME>/Mobile-Price-Predictor.git
cd Mobile-Price-Predictor
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Run the Streamlit Application
Bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501 to use the interactive interface.  
IPYNB

9. Future Roadmap & Enhancements
While the model achieves a strong 92.23% R 
2
  score using foundational hardware parameters, modern smartphone pricing also reflects component-level hardware nuances:  
IPYNB

Processor & Chipset (SoC): Integrating silicon tiers (e.g., Apple A-series, Snapdragon 8-series vs. entry-level MediaTek) to sharpen separation between upper-mid-range and premium flagships.

Display Technology & Refresh Rates: Differentiating between 60Hz LCDs and 120Hz/144Hz LTPO AMOLED displays.

Camera Sensor Specs: Accounting for physical sensor dimensions, Optical Image Stabilization (OIS), and periscope telephoto optics rather than aggregate megapixel count.  
IPYNB

Build Materials & Ingress Protection: Incorporating chassis materials (titanium, aluminum, ceramic) and IP68 water/dust resistance ratings.

Depreciation & Launch Recency: Modeling price decay as a function of months elapsed since product launch.

Because the system was architected using modular Scikit-Learn transformer classes, integrating these additional features requires only registering new transformer steps without refactoring the core inference pipeline.  
IPYNB
