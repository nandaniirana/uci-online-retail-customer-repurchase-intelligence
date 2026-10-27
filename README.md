# UCI Online Retail — Customer Repurchase Intelligence

An end-to-end retail analytics project focused on understanding sales performance, segmenting customers using RFM analysis, and predicting customer repeat-purchase behavior using machine learning. The project combines Python-based analytics, Power BI reporting, and a Streamlit application to translate transactional data into actionable business insights.

## Project Overview

Retail businesses need to understand not only what they sell, but also who their customers are, how customers purchase, and which customers are likely to return.

This project analyzes the UCI Online Retail transactional dataset to explore sales trends, identify customer segments, and develop a predictive approach for repeat purchasing. The resulting analyses are intended to support customer retention, sales planning, and data-driven marketing decisions.

## Business Objectives

- Analyze retail sales performance across time, products, customers, and countries.
- Identify customer purchasing patterns and high-value customer groups.
- Segment customers using Recency, Frequency, and Monetary (RFM) analysis.
- Build and compare machine-learning models for predicting repeat purchases within a 30-day window.
- Evaluate model performance using appropriate classification metrics.
- Present analytical findings through Power BI dashboards and a Streamlit application.
- Translate analytical findings into practical business recommendations.

## Dataset

**Source:** UCI Machine Learning Repository — Online Retail dataset.

The dataset contains transactional records from a UK-based online retailer. Its fields include invoice information, product identifiers and descriptions, quantities, invoice dates, prices, customer identifiers, and countries.

The project uses cleaned transaction data and derived customer-level datasets for analysis and modeling.

Dataset source: https://archive.ics.uci.edu/dataset/352/online+retail

Please review the dataset's terms and attribution requirements before redistributing the original data.

## Project Workflow

### 1. Data Loading and Preparation

- Load the retail transaction data into Python.
- Inspect data types, missing values, duplicates, and data quality.
- Prepare transaction records for analysis.
- Derive revenue-related fields where appropriate.
- Validate the cleaned data before downstream analysis.

### 2. Exploratory Data Analysis

- Analyze sales and revenue trends over time.
- Explore product-level performance.
- Examine customer purchasing behavior.
- Compare sales across countries.
- Use visualizations to identify patterns and potential business opportunities.

### 3. Customer Segmentation Using RFM

RFM analysis evaluates customers across three dimensions:

- **Recency:** How recently a customer purchased.
- **Frequency:** How often a customer purchased.
- **Monetary:** How much revenue a customer generated.

Customer-level RFM features are used for segmentation with K-Means clustering. The resulting segments help distinguish different purchasing behaviors and support targeted retention and engagement strategies.

### 4. Repeat-Purchase Prediction

The predictive modeling workflow investigates whether historical customer behavior can help predict a repeat purchase within a defined 30-day period.

The project includes customer-level features such as:

- Recency
- Tenure
- Purchase frequency
- Total revenue
- Total quantity
- Average order value

Logistic Regression and Random Forest approaches, including balanced variants, are compared as part of the modeling workflow. The evaluation uses classification metrics such as precision, recall, F1-score, and ROC-AUC where appropriate.

Model performance should be interpreted in the context of class imbalance, the prediction-window definition, and the separation of historical features from future outcomes.

### 5. Power BI Reporting

The planned reporting experience organizes the analysis into business-focused views:

- Sales Analysis
- Customer Segmentation
- Repeat-Purchase Prediction
- Business Recommendations

The report is intended to help users explore sales performance, understand customer groups, and interpret the outputs of the predictive analysis.

### 6. Streamlit Application

The Streamlit application provides an interactive interface for exploring the project's analytical outputs. Its current sections include:

- Sales Overview
- Customer Segmentation
- Repeat-Purchase Prediction
- Business Recommendations

The application uses prepared data files and saved prediction outputs. Its displayed predictions should not be interpreted as live predictions for newly entered customers unless that functionality is implemented separately.

## Technology Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Data visualization | Matplotlib, Seaborn, Plotly |
| Machine learning | Scikit-learn |
| Notebook environment | Jupyter Notebook / Google Colab |
| Business intelligence | Microsoft Power BI |
| Interactive application | Streamlit |
| Model and data outputs | Joblib, CSV files |

The final dependency list should match the libraries actually used by the notebook and application.

## Repository Structure

The repository will be organized to separate analysis, application code, data outputs, and documentation.

```text
uci-online-retail-customer-repurchase-intelligence/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── UCI_Online_Retail_Customer_Repurchase_Intelligence.ipynb
│
├── app/
│   └── app.py
│
├── data/
│   └── processed/
│       ├── customer_segments.csv
│       └── customer_predictions.csv
│
├── models/
│   └── saved model files, if required
│
├── powerbi/
│   └── dashboard files and screenshots
│
└── images/
    └── project screenshots
```

This is the target structure. Files will be added only after their actual names, dependencies, and paths have been verified. The application paths may need to be updated to match this layout.

## How to Run the Streamlit Application

### 1. Clone the repository

```bash
git clone https://github.com/nandaniirana/uci-online-retail-customer-repurchase-intelligence.git
cd uci-online-retail-customer-repurchase-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

Once the application file paths and required data files are in place, run:

```bash
streamlit run app/app.py
```

The commands above assume the planned folder structure has been implemented. The README will be updated if the final repository uses a different layout.

## Key Deliverables

- A documented retail data-cleaning and exploratory analysis workflow.
- Customer-level RFM analysis and segmentation.
- A repeat-purchase prediction workflow with model comparison.
- Power BI reporting for sales and customer intelligence.
- A Streamlit interface for exploring analytical outputs.
- Business recommendations derived from the verified analysis.

## Limitations and Considerations

- Historical purchasing behavior does not guarantee future customer behavior.
- Predictive performance depends on feature quality, target definition, and validation methodology.
- Customer segments require business interpretation; cluster labels do not automatically represent fixed customer types.
- Prepared prediction outputs are not equivalent to a continuously updated production prediction service.
- Recommendations should be validated against business context before being used for operational decisions.

## Future Improvements

- Strengthen time-based validation and monitor model performance over time.
- Add segment-specific retention strategies and measurable campaign outcomes.
- Improve prediction explanations and customer-level interpretability.
- Automate data preparation and model evaluation.
- Deploy the application with reproducible dependencies and clear data-loading instructions.

## Author

**Nandani Rana**

