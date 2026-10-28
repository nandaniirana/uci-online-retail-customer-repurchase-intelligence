# UCI Online Retail — Customer Repurchase Intelligence

An end-to-end retail analytics and machine learning project that transforms transactional data into customer intelligence through data cleaning, exploratory data analysis, RFM segmentation, repeat-purchase prediction, and an interactive Streamlit dashboard.

The project combines Python-based analytics with a customer intelligence application to explore sales performance, understand purchasing behavior, and support data-driven retention decisions.

## Project Overview

Retail businesses need to understand what drives sales, how customers behave, and which customers may return to make another purchase.

This project uses the UCI Online Retail dataset to analyze historical transactions, identify customer segments, and investigate whether past purchasing behavior can help predict a repeat purchase within a 30-day window.

The workflow covers transaction data preparation, exploratory analysis, RFM feature engineering, K-Means clustering, predictive model comparison, and interactive visualization using Streamlit.

## Business Objectives

- Analyze revenue and sales trends across time, products, and countries.
- Audit and clean transaction data while validating data quality.
- Identify customer purchasing patterns and high-value customer groups.
- Segment customers using Recency, Frequency, and Monetary (RFM) analysis.
- Build and compare machine learning models for 30-day repeat-purchase prediction.
- Evaluate model performance using classification metrics and threshold analysis.
- Develop an interactive dashboard to explore sales, customer segments, and prediction outputs.
- Translate analytical findings into practical customer retention recommendations.

## Dataset

**Source:** [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)

The dataset contains transactional records from a UK-based online retailer between December 2010 and December 2011.

The original dataset contains 541,909 rows and eight columns, including invoice identifiers, product information, quantities, invoice dates, unit prices, customer identifiers, and countries.

### Dataset summary

| Processing stage | Rows | Columns | Purpose |
|---|---:|---:|---|
| Original dataset | 541,909 | 8 | Initial transaction data |
| Cleaned sales dataset | 524,878 | 9 | Transaction-level sales analysis with revenue |
| Customer-linked transaction records used in modeling preparation | 392,692 | 9 | Customer-level analysis |
| RFM customer dataset | 4,338 | 5 | Customer segmentation |
| Repeat-purchase modeling dataset | 3,965 | 8, including target | Supervised learning |

These figures refer to different stages of the workflow and different eligible populations.

## 1. Data Cleaning and Quality Validation

The initial audit examined missing values, duplicate records, cancellations, transaction quantities, unit prices, and data types.

### Initial data-quality findings

| Quality check | Initial count |
|---|---:|
| Original records | 541,909 |
| Exact duplicate rows beyond the first occurrence | 5,268 |
| Missing product descriptions | 1,454 |
| Missing customer IDs | 135,080 |
| Cancellation invoice rows | 9,288 |
| Negative quantity rows | 10,624 |
| Zero quantity rows | 0 |
| Negative unit-price rows | 2 |
| Zero unit-price rows | 2,515 |
| Missing invoice numbers | 0 |
| Missing invoice dates | 0 |

These categories may overlap and should not be added together to estimate the total number of affected rows.

### Cleaning workflow

1. Inspected dataset dimensions, column names, data types, missing values, and duplicates.
2. Removed exact duplicate transaction rows.
3. Prepared a sales-oriented dataset by excluding non-positive quantities and non-positive unit prices.
4. Calculated transaction revenue using quantity multiplied by unit price.
5. Validated invoice dates and established the observation period.
6. Checked post-cleaning row counts, duplicate records, quantities, unit prices, and revenue totals.
7. Used customer-linked transactions for customer-level segmentation and predictive modeling.

### Post-cleaning results

| Validation metric | Result |
|---|---:|
| Cleaned sales rows | 524,878 |
| Cleaned sales columns | 9 |
| Exact duplicate rows remaining | 0 |
| Non-positive quantities remaining | 0 |
| Non-positive unit prices remaining | 0 |
| Recalculated revenue | £10,642,110.80 |
| Stored revenue total | £10,642,110.80 |
| Revenue reconciliation | Passed |
| Observation period | 1 Dec 2010 – 9 Dec 2011 |

**Missing customer IDs:** The cleaned transaction-level sales dataset still contains 132,186 records with missing customer identifiers. These records were not all removed because transaction-level sales analysis can still use valid transaction information. Customer-level analysis requires identifiable customers and therefore uses customer-linked records.

This distinction is important when comparing overall sales results with customer-level findings.

## 2. Exploratory Data Analysis (EDA)

Exploratory data analysis was performed to understand sales trends, product performance, customer purchasing patterns, and geographic revenue concentration.

### Analyses performed

- Monthly revenue trends
- Product-level revenue and quantity analysis
- Country-level revenue contribution
- Customer revenue and purchasing activity
- Invoice and customer counts
- Sales concentration across products and markets

### Key sales indicators

| Indicator | Result |
|---|---:|
| Total revenue | £10,642,110.80 |
| Units sold | 5,572,420 |
| Unique invoices in cleaned sales data | 19,960 |
| Unique products | 3,922 |
| Countries represented | 38 |
| Average order value reported in the notebook | £533.17 |
| United Kingdom revenue | £9,001,744.09 |
| United Kingdom revenue share | 84.59% |

The average order value is based on the notebook's invoice-level calculation.

### Business interpretation

Revenue is concentrated in the United Kingdom, indicating that market-level analysis should distinguish the primary market from smaller international markets.

Product and customer rankings help identify areas for further investigation. However, revenue alone does not measure profitability; margins, discounts, returns, and operating costs would be needed for a more complete financial assessment.

## 3. Customer Segmentation Using RFM and K-Means

RFM analysis summarizes customer purchasing behavior across three dimensions:

- **Recency:** Days since the customer's most recent purchase.
- **Frequency:** Number of distinct purchase invoices.
- **Monetary:** Total revenue generated by the customer.

The project created an RFM dataset for 4,338 customers and applied K-Means clustering to group customers with similar purchasing characteristics.

### RFM results

| Metric | Result |
|---|---:|
| Customers analyzed | 4,338 |
| Number of clusters | 4 |
| Missing RFM values | 0 |
| Mean recency | 92.54 days |
| Mean frequency | 4.27 invoices |
| Mean monetary value | £2,048.69 |

### Customer distribution by cluster

| Cluster | Customers |
|---|---:|
| 0 | 713 |
| 1 | 1,622 |
| 2 | 837 |
| 3 | 1,166 |
| **Total** | **4,338** |

The resulting customer groups can support further analysis of purchasing frequency, recency, and monetary value.

Cluster numbers are algorithmic identifiers rather than predefined customer types. Business labels such as loyal, high-value, or at-risk should be assigned only after examining the characteristics of each cluster.

## 4. Module 4 — Repeat-Purchase Prediction

This module investigates whether historical customer behavior can help estimate whether a customer will make a qualifying repeat purchase within a subsequent 30-day period.

### Prediction target and data split

| Item | Value |
|---|---|
| Historical cutoff date | 31 Oct 2011 |
| Prediction-window end | 30 Nov 2011 |
| Modeling population | 3,965 customers |
| No repeat purchase in target window | 2,635 (66.46%) |
| Repeat purchase in target window | 1,330 (33.54%) |
| Training customers | 3,172 |
| Test customers | 793 |

The target variable, `RepeatPurchase30D`, indicates whether a customer made a qualifying repeat purchase during the defined future window.

### Features used

- `RecencyDays`
- `TenureDays`
- `Frequency`
- `TotalRevenue`
- `TotalQuantity`
- `AverageOrderValue`

The target was separated from the model input features before training.

### Models evaluated

- Logistic Regression
- Random Forest
- Balanced Logistic Regression
- Balanced Random Forest

The notebook compares models using accuracy, precision, recall, F1-score, ROC-AUC, classification reports, confusion matrices, and probability threshold analysis.

### Baseline model comparison

The following results represent the notebook's default-threshold comparison on the 793-customer test set.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6847 | 0.5284 | 0.5602 | 0.5438 | 0.7156 |
| Random Forest | 0.6948 | 0.5448 | 0.5489 | 0.5468 | 0.7226 |
| Balanced Logistic Regression | 0.6847 | 0.5284 | 0.5602 | 0.5438 | 0.7156 |
| Balanced Random Forest | 0.7188 | 0.6335 | 0.3835 | 0.4778 | 0.7004 |

In this baseline comparison, Random Forest has the highest F1-score among the listed models, at **0.5468**, and the highest ROC-AUC, at **0.7226**.

Balanced Random Forest achieves higher accuracy and precision but lower recall and F1-score. This illustrates why model selection should depend on the business objective rather than accuracy alone.

### Threshold analysis

The notebook also examines a Random Forest classification threshold of 0.30.

| Metric | Result at threshold 0.30 |
|---|---:|
| Accuracy | 0.5233 |
| Precision | 0.3974 |
| Recall | 0.8158 |
| F1-score | 0.5345 |
| ROC-AUC | 0.7226 |

At this threshold, recall increases while precision and accuracy decrease. This operating point may be useful when identifying more potential repeat purchasers is a priority and the business can tolerate additional false positives.

The baseline and threshold-tuned results should be reported separately because they represent different classification thresholds.

### Feature importance

The notebook's Random Forest feature-importance output is:

| Feature | Importance |
|---|---:|
| Frequency | 0.2166 |
| TotalRevenue | 0.2023 |
| TotalQuantity | 0.1946 |
| RecencyDays | 0.1524 |
| TenureDays | 0.1179 |
| AverageOrderValue | 0.1163 |

Frequency has the highest reported importance in this model, followed by total revenue and total quantity.

Feature importance describes the model's relative use of input features. It does not demonstrate that changing a feature will cause a customer to purchase again.

### Modeling considerations

The notebook includes baseline and threshold-tuned evaluations. Further improvements should consolidate the evaluation pipeline, select thresholds using validation data, reserve the test set for final evaluation, and verify that all model features are constructed using information available at the historical cutoff.

## 5. Streamlit Dashboard

The project includes an interactive dashboard titled **UCI Online Retail — Customer Repurchase Intelligence**, developed using Python and Streamlit.

The application uses Pandas for data handling, Plotly for interactive visualizations, and Joblib to load saved model artifacts.

### How the dashboard works

1. The analysis notebook prepares the cleaned transaction data, customer segments, prediction outputs, and model artifacts.
2. These outputs are exported as CSV and Joblib files.
3. `app.py` loads the prepared transaction data, customer segments, saved prediction outputs, and model artifacts.
4. Users navigate the dashboard to explore sales performance, customer groups, prediction probabilities, and business recommendations.

Streamlit caching is used to reduce repeated data and model loading during a session.

### Dashboard sections

#### Sales Overview

Displays:

- Total revenue
- Unique invoices
- Unique customers
- Units sold
- Monthly revenue trend
- Top 10 products by revenue
- Top 10 countries by revenue

Users can filter transaction-based visualizations by invoice date and country.

#### Customer Segmentation

Displays:

- Number of segmented customers
- Customer counts by K-Means cluster
- Average Recency, Frequency, or Monetary value by cluster
- Customer-level segmentation records

#### Repeat-Purchase Prediction

Displays:

- Previously saved Logistic Regression or Random Forest probabilities
- Number of customers scored
- Average predicted probability
- Count of customers with predicted probability at or above 60%
- Distribution of predicted probabilities
- Customer-level prediction records sorted by probability

**Implementation detail:** The application displays previously generated prediction outputs. Although it loads saved model artifacts, the current prediction page does not generate live predictions for newly entered customers.

#### Business Recommendations

Generates contextual prompts based on the selected transaction data, including:

- Reviewing high-revenue products
- Examining the strongest market by revenue
- Considering retention campaigns for customers with higher predicted probabilities
- Reviewing profitability before expanding promotions
- Tailoring customer engagement to verified RFM cluster profiles

These recommendations are decision-support prompts, not causal findings. Revenue is not the same as profit, and predicted probabilities are estimates rather than guarantees.

### Required application files

The current `app.py` expects these files in the same directory as the application:

```text
app.py
transactions.csv.gz
customer_segments.csv
customer_predictions.csv
logistic_regression.joblib
random_forest.joblib
```

The local project folder also contains `config.json`; its purpose should be verified before deciding whether to include it in the public repository.

The application paths and filenames must match the final repository structure, and all required files must be available when the dashboard runs.

## 6. Technology Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine learning | Scikit-learn |
| Model serialization | Joblib |
| Dashboard framework | Streamlit |
| Notebook environment | Google Colab / Jupyter Notebook |
| Version control | Git and GitHub |

The final `requirements.txt` should reflect the dependencies actually used by the notebook and dashboard.

## 7. Run the Streamlit Dashboard Locally

### Prerequisites

- Python installed
- Required CSV and model files available
- Dependencies installed from `requirements.txt`

### Clone the repository

```bash
git clone https://github.com/nandaniirana/uci-online-retail-customer-repurchase-intelligence.git
cd uci-online-retail-customer-repurchase-intelligence
```

### Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

If the repository follows the intended structure:

```bash
streamlit run app/app.py
```

The required CSV and model files must be placed beside `app.py`, as expected by the current code. The command and paths should be verified against the final repository before publication.

## 9. Limitations and Future Improvements

### Current limitations

- Historical purchasing behavior does not guarantee future purchases.
- A substantial portion of original transactions lack customer IDs.
- Sales-level and customer-level analyses use different eligible populations.
- Cluster numbers require business interpretation.
- Prediction thresholds change precision and recall.
- The current Streamlit prediction page displays saved outputs rather than live predictions for new customers.
- Revenue-based recommendations do not replace margin and profitability analysis.

### Future improvements

- Consolidate the notebook and remove duplicated diagnostic or export cells.
- Strengthen time-based validation and confirm feature availability at the prediction cutoff.
- Select thresholds on validation data and report final test metrics consistently.
- Add prediction explanations and more detailed customer segment profiles.
- Measure retention campaigns through controlled experiments.
- Automate data preparation, validation, and model evaluation.
- Add dashboard screenshots and deployment documentation.
- Improve reproducibility through a verified dependency file.

## 10. Author

**Nandani Rana**
MCA | Software Engineering | Data Analytics and Machine Learning
---

*This project demonstrates an applied workflow for retail data preparation, customer analytics, predictive modeling, and interactive dashboard development. Results are specific to the dataset, observation period, feature definitions, and evaluation setup described above.*
