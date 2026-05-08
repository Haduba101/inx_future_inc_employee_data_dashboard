# INX Future Inc. Employee Data Dashboard

A Streamlit web application for visualizing and analyzing employee data from INX Future Inc. This dashboard analyzes employee data and supports hiring and HR decisions by predicting employee performance ratings.

## Features

- **Data Overview**: Summary statistics including total employees, departments, job roles, and attrition rate
- **Attrition Analysis**: Visualize attrition counts by department
- **Department Analysis**: Employee distribution across departments with pie chart
- **Age Distribution**: Histogram of employee ages
- **Salary Analysis**: Box plot of hourly rates by department
- **Interactive Filters**: Filter data by department, job role, and attrition status

## Installation

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd INX
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   # Activate the virtual environment
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Notebook Analysis

- `inx_data.ipynb`: Jupyter notebook with in-depth exploratory data analysis and predictive modeling.
- Objective: analyze employee data and build a model to support hiring and HR decisions by predicting employee performance ratings.
- Includes 10+ visualizations, correlation analysis, outlier detection, and model training for Logistic Regression, Random Forest, and XGBoost.
- The notebook saves the production-ready XGBoost pipeline using `joblib` as `xgb_model_pipeline.pkl` or `xgb_model_pipeline.joblib`.

## Data

- `inx_data.csv`: Main employee dataset containing various attributes like age, department, job role, salary, performance rating, and HR-related factors.
- `data_definitions.csv`: Definitions for categorical variables (e.g., education levels, satisfaction ratings)

## Libraries

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib
