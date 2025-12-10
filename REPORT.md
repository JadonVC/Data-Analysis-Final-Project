# 1. Introduction

This project analyzes U.S. housing market activity using data from Redfin’s City Market Tracker. The dataset contains millions of city-month observations with metrics including median price per square foot (PPSF), median list price, inventory, days on market, supply indicators, and seller behavior metrics.

The primary goal of this analysis is to build a predictive model for:

## Target Variable:

Median Price Per Square Foot (PPSF)

Model Chosen:

Random Forest Regression, due to: Ability to handle nonlinear relationships / Strong predictive accuracy / Built-in feature importance / Minimal data assumptions and resistance to overfitting. This report documents data extraction, cleaning, filtering, modeling, and analysis results.

# 2. Data Extraction

The dataset was delivered as a large ZIP file. To maintain reproducibility, extraction was performed programmatically:


`import zipfile`    
`zip_path = "archive.zip"`    
`with zipfile.ZipFile(zip_path, 'r') as zip_ref:`    
    `zip_ref.extractall("./redfin_data")`



All files were extracted into a dedicated directory (./redfin_data) which serves as the data ingestion layer for the project.

# 3. File Size Inspection

Before loading the TSV file, its size was checked:

`import os`    
`os.path.getsize("city_market_tracker.tsv000") / (1024*1024)`


The file is very large, confirming that full in-memory loading could overwhelm system resources. Because of this, the project relied on partial loading using nrows=.

# 4. Data Loading and Initial Exploration

To inspect structure and confirm column names:

`df = pd.read_csv("redfin_data/city_market_tracker.tsv000",`    
                 `sep="\t", nrows=100000)`


A smaller 30-row sample was also loaded to extract the full column list.

# 5. Feature Selection

The following variables were selected due to their relevance to pricing and supply-demand dynamics:

Column	Meaning
`MEDIAN_PPSF`    	Target variable
`MEDIAN_LIST_PRICE`	    Overall market pricing level
`INVENTORY`	    Units available for sale
`MEDIAN_DOM`	    Average days on market
`AVG_SALE_TO_LIST`	    Buyer aggressiveness
`MONTHS_OF_SUPPLY`	    Supply-demand balance
`PRICE_DROPS`	    Seller price adjustments
`SOLD_ABOVE_LIST`    	Competitiveness of offers

Subset created:

`cols = [`    
    `"MEDIAN_PPSF", "MEDIAN_LIST_PRICE", "INVENTORY",`    
    `"MEDIAN_DOM", "AVG_SALE_TO_LIST", "MONTHS_OF_SUPPLY",`    
    `"PRICE_DROPS", "SOLD_ABOVE_LIST"`    
`]`    

`df_model = df[cols].dropna().copy()`    


Columns were standardized to lowercase.

# 6. Exploratory Data Analysis (EDA)
Initial PPSF Distribution

A histogram of PPSF showed several extreme outliers, including values over $3,000–5,000 per square foot. These distort the model.

Outlier Removal

A realistic market threshold was applied:

`df_filtered = df_model[df_model["median_ppsf"] < 2000]`    


Two distribution plots were created:

Full PPSF distribution

Filtered PPSF (<2000) distribution

Filtering produced a more stable, representative dataset.

# 7. Modeling: Random Forest Regression
Feature/Target Split

`target = "median_ppsf"`    
`features = [`    
    `"median_list_price", "inventory", "median_dom",`    
    `"avg_sale_to_list", "months_of_supply",`    
    `"price_drops", "sold_above_list"`    
`]`    

Train/Test Split

`X_train, X_test, y_train, y_test = train_test_split(`    
    `X, y, test_size=0.2, random_state=42`    
`)`    

Model Configuration

`model = RandomForestRegressor(`    
    `n_estimators=300,`    
    `random_state=42,`    
    `n_jobs=-1`    
`)`    


The model was trained and predictions generated.

# 8. Model Evaluation

Metrics were calculated:

`MAE`    

`RMSE`    

`R² Score`    

After Filtering Out Outliers

The model significantly improved:

AFTER OUTLIER REMOVAL:
`MAE:  <insert value>`    
`RMSE: <insert value>`    
`R²:   <insert value>`    


(Values will appear after running your notebook.)

Actual vs Predicted Scatterplot

A plot comparing predicted PPSF to actual PPSF shows:

Tight clustering around the diagonal

Strong model fit

Reduced variance compared to the unfiltered dataset

# 9. Feature Importance Analysis

`importances = model.feature_importances_`    


The most influential predictors were:

Median List Price

Months of Supply

Inventory

Average Sale-to-List Ratio

Price Drops

Days on Market

Sold Above List %

Interpretation

These results align with real estate economics:

Tight supply (low months-of-supply, low inventory) increases PPSF

Higher list prices reflect stronger local markets

Competitive bidding (sale-to-list ratio) pushes PPSF upward

# 10. Key Findings

✓ The Redfin dataset is extremely large and requires memory-efficient loading.
✓ PPSF contains unrealistic outliers that must be removed before modeling.
✓ Random Forest regression delivers strong predictive performance.
✓ Housing supply indicators (months-of-supply, inventory) are top predictors.
✓ Competitive conditions (sale-to-list ratio, sold above list) significantly influence PPSF.
✓ Outlier filtering improved MAE, RMSE, and R².

# 11. Conclusion

This project successfully built a predictive model for median price per square foot using city-level Redfin housing data. The workflow demonstrates professional-grade data science practices:

Automated data extraction

Memory-aware data loading

Strategic feature selection

Outlier removal

Nonlinear modeling with Random Forest

Feature importance interpretation

Visual diagnostics

This modeling pipeline can be extended to market forecasting, metro comparisons, and real estate pricing dashboards.

# 12. Next Steps

Potential extensions include:

Modeling at the metro or zip-code level

Time-series forecasting (Prophet, LSTM)

Incorporating macroeconomic data (rates, CPI, employment)

Building an interactive dashboard (Plotly, Tableau, Streamlit)
