Mobile-Price-Predictor
---------------------------------
End-to-end smartphone price prediction pipeline featuring custom feature engineering, log-transformed Ridge regression (~92% R²), and an interactive Streamlit web app.

<br>

1. Data Cleaning
---------------------------------------------------
The raw dataset contained 407 records with embedded measurement units, irregular whitespace, and non-numeric formats across key hardware columns. Data cleaning began by standardizing column headers and using regular expressions to extract pure numeric values for RAM, storage capacity, and screen size. Dollar signs and commas were stripped from the price column before casting it to an integer, while complex multi-lens camera strings (e.g., "108 + 10 + 10 + 12") were parsed to calculate the total number of rear camera lenses. Finally, 49 duplicate rows were identified and removed, producing a clean dataset of 358 unique smartphone configurations ready for modeling.

<br>

2. Deterministic Train/Test Split (Hash ID):
-----------------------------------------------------
To prevent data leakage and guarantee reproducibility without relying on arbitrary random seeds, a composite string identifier was created from Brand, Model, Storage (GB), and RAM (GB). Each identifier was hashed using the CRC32 algorithm, assigning records with hashes below 20% of $2^{32}$ to the test set. This secured an immutable split of 300 training samples and 58 hold-out test samples across all runs.

<br>

3. Custom Feature Engineering
------------------------------------------------------------
Three custom Scikit-Learn transformers were developed to incorporate commercial domain knowledge: BrandValueAdder for brand equity scoring, ServiceScoreAdder for after-sales support reputation, and SpecTierClassifier to segment devices into Budget, Mid-Range, or Flagship tiers. These fed into an automated ColumnTransformer handling numerical scaling and categorical one-hot encoding, expanding the input space to 25 engineered features.

<br>

4. Final Model Performance & Metrics:
-----------------------------------------------------------------------

<img src="Mobile Price Predictor/Pictures/final_model_scores.png" alt="Final Model Scores" width="500"/>

The final model wrapped an $L_2$-regularized Ridge regressor inside TransformedTargetRegressor using log-scale functions (log1p and expm1) to handle right-skewed pricing. Evaluated on the unseen test set, it delivered an $R^2$ score of 92.23%, an RMSE of $93.63, and a Mean Absolute Error of $59.39.

<br>

5. Gradient Descent from Scratch:
---------------------------------------------------------------------

<img src="Mobile Price Predictor/Pictures/batch_gradient_descent.png" alt="Batch Gradient Descent" width="500"/>

To mathematically validate Scikit-Learn's optimization, a custom Batch Gradient Descent algorithm was built from scratch using pure NumPy matrix vectorization. Running for 2,000 iterations, the manual implementation converged to a 92.31% $R^2$ score and $93.14 RMSE, confirming theoretical alignment.

<br>

6. Pipeline Serialization (Joblib):
-----------------------------------------------
The complete end-to-end workflow—spanning custom market scoring transformers, one-hot encoders, standard scalers, and the regularized log regressor—was unified and serialized into mobile_price_pipeline.pkl using Joblib. This produced a standalone artifact that ingests raw, unformatted phone specifications and returns instant dollar predictions without manual data preparation.

<br>
     
8. Interactive Web App (Streamlit):
 ----------------------------------------------------  
 
<img src="Mobile Price Predictor/Pictures/streamlit_web_app.png" alt="Streamlit Web App" width="750"/>

Live Link - https://mobile-price-predictor-bnwuljvyrdkzct4begr7mk.streamlit.app/

An intuitive web interface was developed in app.py using Streamlit to serve the serialized model for real-time user interaction. The application accepts user-selected hardware specifications, outputs the estimated price alongside a 95% confidence interval derived from test MAE, and dynamically assigns market tier badges (Budget, Mid-Range, or Flagship). 

<br>

8. Future Roadmap & Enhancements:
-----------------------------------------------------------
While the current model captures over 92% of market price variance, predictions can be sharpened further by incorporating deeper hardware attributes such as Processor/SoC tier, display refresh rate (60Hz vs. 120Hz LTPO), and optical image stabilization (OIS). Because the project uses modular Scikit-Learn transformers, these new attributes can be plugged in seamlessly without altering downstream deployment code. 
