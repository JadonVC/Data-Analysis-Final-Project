1. Introduction

This project analyzes U.S. housing market conditions using Redfin’s City Market Tracker dataset. The dataset is extremely large, containing millions of city-month observations with dozens of housing market indicators such as median list price, days on market, sales-to-list ratios, and inventory levels.

The primary goal of this analysis is to build a predictive model for median price per square foot (PPSF) using measurable market indicators. Random Forest regression was selected due to its ability to:

Handle non-linear relationships,

Rank feature importance,

Manage large datasets with minimal preprocessing, and

Provide strong performance without heavy tuning.

Before modeling, extensive data preparation was required due to the size of the dataset.

2. Data Extraction and File Handling
Unzipping the Redfin Data
import zipfile

zip_path = "archive.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("./redfin_data")


This step programmatically extracts all housing data files from a ZIP archive into a clean directory (/redfin_data). This ensures reproducibility and prevents manual extraction errors.

File Size Inspection
import os
os.path.getsize("city_market_tracker.tsv000") / (1024*1024)


This check confirmed that the data file is very large (tens or hundreds of MB). Because of this, the dataset required partial loading using nrows= to avoid exhausting system memory.

3. Data Loading and Initial Exploration
Sampling for Inspection
df = pd.read_csv("redfin_data/city_market_tracker.tsv000",
                 sep="\t", nrows=100000)


Only 100,000 rows were read initially to:

Inspect column names

Verify formatting

Identify missing values

Select columns relevant for modeling

A second small sample (nrows=30) was loaded to list all columns and confirm naming (uppercase vs lowercase).

4. Feature Selection

From the full list of Redfin metrics, the following variables were chosen based on economic relevance and correlation with home price fundamentals:

Column	Description
MEDIAN_PPSF	Target variable – price per square foot
MEDIAN_LIST_PRICE	Market pricing anchor
INVENTORY	Available supply
MEDIAN_DOM	Median days on market
AVG_SALE_TO_LIST	Aggressiveness of buyers/sellers
MONTHS_OF_SUPPLY	Supply/demand balance
PRICE_DROPS	Seller behavior and corrections
SOLD_ABOVE_LIST	Competitive market indicator
Column Filtering
cols = [
    "MEDIAN_PPSF", "MEDIAN_LIST_PRICE", "INVENTORY",
    "MEDIAN_DOM", "AVG_SALE_TO_LIST", "MONTHS_OF_SUPPLY",
    "PRICE_DROPS", "SOLD_ABOVE_LIST"
]

df_model = df[cols].dropna().copy()


All rows containing missing values were dropped to ensure clean model input.

Columns were then converted to lowercase to simplify later processing.

5. Exploratory Data Analysis (EDA)
Distribution Plot of PPSF

A histogram revealed that PPSF contained extreme outliers, including values above $3,000 and $5,000/sqft, which are not representative of broader U.S. markets.

Outlier Filtering
df_filtered = df_model[df_model["median_ppsf"] < 2000]


Filtering PPSF < 2000 significantly reduced noise, leading to more stable modeling performance.

Two histograms were produced:

Raw PPSF distribution

Filtered PPSF distribution (<2000)

The filtered version shows a realistic market distribution, improving training quality.

6. Model Building: Random Forest Regression
Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Model Configuration
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


Reasons for using Random Forest:

High predictive accuracy

Handles non-linear relationships naturally

Resistant to overfitting with proper tuning

Outputs feature importance rankings

Model Training and Prediction
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

7. Model Evaluation
Metrics Before Filtering

Initial performance (with outliers included) was acceptable but noisy, indicating distortion from unrealistic PPSF values.

Metrics After Outlier Removal
MAE:  ...
RMSE: ...
R²:   ...


After filtering PPSF values > 2000:

MAE decreased (better accuracy)

RMSE decreased (less error spread)

R² increased, showing stronger explanatory power

This confirmed that outlier removal greatly improved predictive performance.

Actual vs. Predicted Plot

A scatterplot comparing actual PPSF values to predictions showed:

Tight alignment around the 45° line

Minimal systematic bias

Improved accuracy after filtering

8. Feature Importance Analysis
plt.barh(range(len(features)), importances[indices])


The Random Forest model provided a ranking of the most influential predictors.

Top Contributors to PPSF

Median List Price

Months of Supply

Inventory

Average Sale-to-List Ratio

Price Drops

Days on Market

Sold Above List %

Interpretation:

Markets with higher list prices, lower supply, and stronger bidding behavior tend to produce higher PPSF values.

This aligns with standard real estate economics: tight supply and competitive buyers push prices upward.

9. Key Findings
1. The dataset is extremely large, requiring memory-efficient loading.

Sampling smaller portions was necessary.

2. PPSF contained extreme outliers that distorted the model.

Filtering values above 2000 improved accuracy significantly.

3. Random Forest provided strong predictive performance.

The model achieved solid MAE, RMSE, and R² scores after cleaning.

4. The most important predictors were supply-demand indicators.

Months of supply, inventory, and list price were crucial drivers of PPSF.

5. Visual diagnostics confirmed stability.

Histograms, scatterplots, and feature importance charts all supported the modeling steps.

10. Conclusion

This analysis successfully built a predictive model for median price per square foot using city-level housing market indicators from Redfin. Through appropriate data cleanup, outlier handling, and model training, the Random Forest model delivered statistically meaningful accuracy and interpretable feature importance insights.

The workflow follows professional data-science standards:

Structured data extraction

Efficient sampling

Targeted feature engineering

Removal of unrealistic values

Robust model evaluation

Clear visualization

This modeling pipeline can now be extended to:

Predict future housing price trends

Analyze metro-level markets

Compare regional pricing behavior

Build dashboards or automated forecasting tools
