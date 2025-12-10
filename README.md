# Data-Analysis-Final-Project
Collaborative final project for data analysis class by Jadon Chanthavong / Sairam Veerasurla / Cameron Wardle

# Housing Market Analysis: Regional Price Variation and Infrastructure Impact

## Introduction

This project uses data science and machine learning techniques to analyze how housing prices vary across the United States. The main focus is on how **location, price per square foot, and basic market indicators** (inventory, days on market, etc.) relate to each other.

We started with three main questions:

1. **Does square footage have a “standard” price per square foot across the U.S., or does it vary a lot by region?**  
2. **Do infrastructure and population density (big metros vs. less-developed areas) affect price per square foot in surrounding areas?**  
3. **Are there natural “market segments” in the U.S. housing market (e.g., premium, mid-range, affordable), and how many distinct market types exist?**

The overall goal of the project is to understand how housing prices behave across states and cities, and to show that the U.S. housing market is not one unified system but a collection of different market tiers that depend on local conditions.

---

## About the Data

### Source

- **Dataset:** US Cities Housing Market Data  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/vincentvaseghi/us-cities-housing-market-data/  
- **File used:** `city_market_tracker.tsv000` (tab-separated)

This dataset contains housing statistics for thousands of U.S. cities over time. It combines **pricing, inventory, and transaction metrics** that allow us to study both price levels and market activity across regions.

### Structure and Features

From a 50,000-row sample:

- **Rows:** 50,000 (before cleaning)  
- **Columns:** 58  
- **Memory usage:** ~46.8 MB  

Important fields used in the analysis:

- **Location:** `CITY`, `STATE`, `STATE_CODE`, `PARENT_METRO_REGION`
- **Pricing:**  
  - `MEDIAN_SALE_PRICE`  
  - `MEDIAN_PPSF` (median price per square foot)  
  - `MEDIAN_LIST_PRICE`, `MEDIAN_LIST_PPSF`
- **Market activity:**  
  - `HOMES_SOLD`, `PENDING_SALES`, `NEW_LISTINGS`, `INVENTORY`  
  - `MONTHS_OF_SUPPLY`, `MEDIAN_DOM` (days on market)  
  - `AVG_SALE_TO_LIST`, `PRICE_DROPS`, `OFF_MARKET_IN_TWO_WEEKS`
- **Property characteristics:**  
  - `PROPERTY_TYPE`, `PROPERTY_TYPE_ID`

### Missing Values and Basic Stats

- **Total missing values:** 443,915 (~15.3% of all cells)  
- **Columns with missing data:** 42 (out of 58)  
- The worst columns include `PRICE_DROPS`, `PRICE_DROPS_MOM`, `PRICE_DROPS_YOY`, and some YOY/MOM rate columns.

Key statistics (before heavy cleaning):

- **Median sale price (mean):** \$320,923  
- **Median sale price range:** \$585 – \$29,000,000  
- **Price per square foot (mean):** \$185  
- **Price per square foot range:** \$0 – \$145,115  
- **Coefficient of variation (PPSF):** ~612% (extremely skewed due to outliers)

Geographic coverage:

- **Unique states:** 51  
- **Unique cities:** 12,152  

### Exploratory Visualizations

Several plots are generated and saved as images:

- `housing_analysis.png` – Histograms of sale price and price per sqft, top states by price, and property type counts.  
- `housing_knn_analysis.png` –  
  - Scatterplot of `MEDIAN_PPSF` vs `MEDIAN_SALE_PRICE`  
  - Boxplot of price by property type  
  - Heatmap of average price per sqft by state  
  - Distribution of median sale prices (log scale)

These visualizations helped confirm that:

- Prices and price per sqft are **heavily skewed** and driven by extreme outliers.  
- There are clear differences in average pricing across states.  
- Property types have distinct price distributions.

---

## Methods

### Pre-processing

We performed a multi-step cleaning process to make the dataset usable for analysis:

1. **Load data**  
   - Loaded 50,000 rows from `city_market_tracker.tsv000` with `sep='\t'`.

2. **Remove rows with missing critical values**  
   - Required columns: `MEDIAN_SALE_PRICE`, `MEDIAN_PPSF`, `STATE`, `CITY`.  
   - After dropping rows with nulls in these columns, records dropped from 50,000 to **49,318**.

3. **Remove zero / invalid prices**  
   - Removed rows where `MEDIAN_SALE_PRICE <= 0` or `MEDIAN_PPSF <= 0`.  
   - In the subset used for most analyses, this did not remove many additional rows (data was mostly positive already).

4. **Remove extreme outliers using percentiles**  
   - For `MEDIAN_SALE_PRICE`: kept the **5th to 95th percentiles** (middle 90%):  
     - Range roughly \$55,878 – \$827,575.  
   - For `MEDIAN_PPSF`: kept the **5th to 95th percentiles**:  
     - Range roughly \$39 – \$434.  
   - After trimming outliers, records dropped to **42,781** for many analyses and **44,386** for others, depending on the exact filters.

5. **Data quality check**

   After cleaning:
   - **Records retained:** ~42,781 (about **85.6%** of original 50,000)  
   - **Average sale price:** ~\$266,173  
   - **Average price per sqft:** ~\$153  
   - **PPSF coefficient of variation:** ~52.1% (still high, but much more realistic)

6. **Geographic and property coverage**

   - **States:** 51  
   - **Cities:** ~11,220  
   - **Property types:** 5 main categories  
     - All Residential (~38%)  
     - Single Family Residential (~38%)  
     - Condo/Co-op (~10%)  
     - Townhouse (~9%)  
     - Multi-Family (2–4 Unit) (~5%)

This preprocessing made the dataset realistic enough for both **variation analysis** and **clustering**, while still preserving most of the data.

---

### Method 1: Price Per Sqft Variation (Coefficient of Variation)

To answer whether square footage has a consistent price per sqft nationally, we:

- Grouped the dataset by:
  - **State**, and
  - **Property type**, and
  - **State + property type** combinations.
- Computed for each group:
  - Mean `MEDIAN_PPSF`
  - Standard deviation of `MEDIAN_PPSF`
  - **Coefficient of variation (CV) = std / mean * 100**

Interpretation:

- **High CV** → prices per sqft are very inconsistent in that group.  
- **Low CV** → prices per sqft are more uniform in that group.

We used this to identify:

- The states with the **highest** and **lowest** pricing volatility.  
- The property types with the highest volatility.  
- Specific (state, property type) combinations with extreme variation.

### Method 2: Market Segment Identification Through K-means Clustering

To study natural market segments, we used **K-means clustering** on engineered features.

Steps:

1. **Feature selection**

   Selected numeric and encoded features including:

   - `MEDIAN_PPSF`  
   - `MEDIAN_SALE_PRICE`  
   - Encoded `STATE` (`STATE_ENCODED`)  
   - Encoded `PROPERTY_TYPE` (`PROPERTY_TYPE_ENCODED`)  
   - Market metrics (filled with medians if missing):  
     - `HOMES_SOLD`, `INVENTORY`, `MONTHS_OF_SUPPLY`, `MEDIAN_DOM`

2. **Encoding and scaling**

   - Encoded string columns with `LabelEncoder`.  
   - Standardized all features using `StandardScaler` for K-means.

3. **Choosing number of clusters (k)**

   - Tested k from **2 to 10**.  
   - For each k, computed:
     - **Silhouette score** (how separated the clusters are)  
     - **Inertia** (within-cluster sum of squares)
   - Silhouette scores were in the ~0.22–0.26 range.  
   - Highest scores occurred at **k = 8 and k = 10 (0.257)**.  
   - We chose **k = 10** to capture more nuance, but also discuss the more interpretable **5–cluster view**.

4. **Final clustering**

   - Fit K-means with k = 10 on the scaled features.  
   - Added a `CLUSTER` label for each record.  
   - Calculated, per cluster:
     - Average sale price  
     - Average price per sqft  
     - Price range  
     - Top states and property types  
     - PPSF variation within cluster

5. **Visualization**

   - Saved `clustering_analysis.png` showing:
     - Silhouette score vs. k  
     - Cluster size distribution  
     - Average sale price by cluster  
     - Average price per sqft by cluster

---

## Evaluation

### 1. Variation in Price Per Sqft

Key results:

- **Overall PPSF coefficient of variation:** ~54.0%  
  → Strong evidence that price per square foot is *not* consistent across regions.

- **Most volatile states by PPSF (highest CV):**
  - New York: ~59.0% (mean ~\$166/sqft)  
  - Wyoming: ~55.5%  
  - South Carolina: ~54.1%  
  - New Mexico: ~53.8%  
  - Florida: ~50.6%

- **Most stable states by PPSF (lowest CV):**
  - Hawaii: ~33.3% (mean ~\$293/sqft)  
  - Alaska, Delaware, Maryland, Massachusetts also show relatively low CV (~36–41%).

- **Property type volatility:**
  - Multi-Family (2–4 Unit): ~63.3% CV (most volatile)  
  - Townhouse: ~47.5% CV (least volatile of the group)  
  - Single Family and All Residential: ~54% CV

These metrics directly evaluate how consistent or inconsistent pricing is within each region and property type.

### 2. Clustering Evaluation

For the clustering model:

- **Silhouette scores (k=2–10):**
  - Range from ~0.217 to ~0.257  
  - Best scores at **k = 8 and 10 (0.257)**  
- **Final k selected:** 10 clusters  
- **Records used for clustering:** ~44,386 after cleaning.

Cluster examples:

- **Cluster 0 – Premium Market (≈12% of records):**
  - Avg sale price: ~\$606K  
  - Avg PPSF: ~\$315  
  - Top states: California, Massachusetts, New York  
  - Low volatility (~39% CV) → stable premium markets.

- **Cluster 1 / 2 / 3 / 5 – Core market tiers (~74% of records combined):**
  - Avg price range: ~\$218K–\$231K  
  - Avg PPSF: ~\$130–\$145  
  - Represent “typical” U.S. housing markets.

- **Cluster 5 – Affordable Tier:**
  - Avg sale price: ~\$224K  
  - Avg PPSF: ~\$130  
  - Heavily represented by Pennsylvania, New York, Texas.

The evaluation for clustering is based on:

- Silhouette score trends  
- How well clusters align with intuitive “premium / mid-range / affordable” tiers  
- Stability within each cluster (price variation, CV)  
- Geographic distribution of states across clusters

---

## Storytelling and Conclusion

### Did square footage have a universal national price?

No. The **54% overall variation** in price per square foot and the large differences between states (e.g., Hawaii vs. low-cost regions) show that there is **no single national rate** for housing space. A square foot of housing is simply worth far more in some regions than others.

### How do infrastructure and population density matter?

States do **not** cluster neatly by geography. Instead:

- California appears across multiple tiers: premium, mid-range, and more affordable clusters.
- New York also spans premium, corridor, and affordable segments.
- High-volatility states like Florida and Illinois include both dense urban markets (Miami, Chicago) and much cheaper rural markets.

This pattern supports the story that **local infrastructure and population density** (coastal metros, transit corridors, job centers) drive prices, not just state boundaries.

### Are there natural market segments?

Yes. Between **5 and 10 distinct market segments** emerge from the clustering:

- A clear **premium/luxury segment** with very high prices and relatively stable pricing.
- Several **mid-range and typical segments** that cover most of the U.S. market.
- An **affordable tier** with lower prices but still wide variation.
- A few **niche micro-segments** (very small clusters) representing special conditions or edge cases.

Overall, the project shows that the U.S. housing market behaves like a set of **stacked tiers** rather than one smooth national market. Pricing is heavily influenced by **location, infrastructure, and property type**.

### What I learned (project + class)

Through this project and the course, I learned how to:

- Work with a **large, messy real-world dataset**, including cleaning, handling missing values, and trimming outliers.
- Use **summary statistics and visualizations** to understand a dataset before modeling.
- Apply **K-means clustering** and interpret silhouette scores and inertia to pick a reasonable k.
- Use **coefficient of variation** as a way to compare variability across different groups.
- Turn raw outputs (tables, metrics, cluster summaries) into a **story** that answers the original questions and connects back to real-world housing markets.

---

## Impact

Even though this is a class project, the ideas behind it have real-world impact.

### Potential Positive Impacts

1. **Better investment decisions**  
   Investors and developers can use market segmentation (premium, mid-range, affordable, etc.) to choose markets that match their risk and return goals. Stable premium markets might be attractive for long-term holds, while high-volatility markets might appeal to more risk-tolerant investors.

2. **Improved pricing and valuation**  
   Real estate agents and appraisers could use insights about how strongly location and infrastructure affect price per sqft to produce more accurate valuations and comps, reducing both overpricing and underpricing.

3. **Smarter policy and infrastructure planning**  
   Planners could see where infrastructure investments might have the highest payoff. For example, if infrastructure-rich corridors show stable premium pricing, similar investments in underdeveloped areas might help stabilize those markets.

4. **Reduced information asymmetry**  
   Many buyers and sellers think in “national average” terms. Showing that markets break into multiple tiers can help people understand that location and local conditions matter more than national headlines.

### Potential Negative Impacts

1. **Predatory lending or targeting**  
   Lenders could use knowledge of high-volatility segments to aggressively market risky loans in vulnerable markets (e.g., volatile multifamily segments), repeating past patterns of predatory lending.

2. **Reinforcing inequality**  
   If policymakers only invest in already stable, premium markets because they look “safe” on paper, they could neglect volatile or affordable regions and widen the gap between wealthy and struggling areas.

3. **Neighborhood discrimination and steering**  
   Agents or investors might avoid areas labeled as “high-volatility” or “affordable tier,” reinforcing segregation or limiting mobility, even if some neighborhoods within those segments are improving.

4. **Oversimplification of local reality**  
   Clusters and segments are averages. If people rely too heavily on these labels, they might ignore critical local details like school quality, safety, or upcoming development, leading to bad decisions based on incomplete information.

Overall, the project shows that **data-driven market segmentation is powerful**, but it must be used carefully and ethically to avoid harming the very communities it describes.

---

## Repository, Code, and Data Access

This repository should include:

- `Final.ipynb` – main notebook containing:
  - Data exploration  
  - Preprocessing  
  - Variation analysis (CV)  
  - Clustering analysis  
  - Printed outputs and key metrics
- `data/city_market_tracker.tsv000` – the main data file (if file size is too large for GitHub, include instructions below instead of the file).
- `images/` folder containing:
  - `housing_analysis.png`  
  - `housing_knn_analysis.png`  
  - `ppsf_variation_analysis.png`  
  - `clustering_analysis.png`

### How to Run

1. Download the dataset from Kaggle  
   - Go to: https://www.kaggle.com/datasets/vincentvaseghi/us-cities-housing-market-data/  
   - Download the data and place `city_market_tracker.tsv000` into a local `data/` folder (or the root, matching how the notebook reads it).

2. Open `Final.ipynb` in Jupyter / VS Code.

3. Run all cells in order.

4. Generated images will be saved in the working directory and can be viewed or uploaded to GitHub.

---
