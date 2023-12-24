# CSV Analysis and Regression Model Training App

This Streamlit app enables interactive analysis and preprocessing of CSV datasets. Users can explore data, handle missing values, drop columns, and train regression models (Linear Regression or Random Forest). The app also allows users to download the preprocessed dataset and the trained regression model.

## Features

- **Upload and Explore**: Upload a CSV dataset and explore its first 10 records, correlation matrix, and missing values.
- **Data Preprocessing**: Handle missing values by choosing mean, median, mode, or dropping records. Drop unnecessary columns.
- **Regression Model Training**: Select columns for features (X) and a target column for the regression model (Y). Train models such as Linear Regression or Random Forest.
- **Download Results**: Download the updated dataset and the trained regression model.

## Usage

1. Install the required dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. Upload a CSV file and interact with the app.

or you can click on the link here - https://dataset-analysis-and-model-training-app.streamlit.app/

## Requirements

- Python 3.7 or higher
- Streamlit
- Pandas
- Seaborn
- Scikit-learn
- Joblib
