import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def load_data(file):
    df = pd.read_csv(file)
    st.subheader("1. Show first 10 records of the dataset")
    st.dataframe(df.head(10))

    return df

def show_correlation(df):
    st.subheader("2. Show the correlation matrix and heatmap")
    numeric_columns = df.select_dtypes(include=['number']).columns
    correlation_matrix = df[numeric_columns].corr()
    st.dataframe(correlation_matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    st.pyplot(fig)

def show_missing_values(df):
    st.subheader("3. Show the number of missing values in each column")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values)

def handle_missing_values(df):
    st.subheader("4. Handle missing values")
    numeric_columns = df.select_dtypes(include=['number']).columns

    fill_option = st.radio("Choose a method to handle missing values:", ('Mean', 'Median', 'Mode', 'Drop'))
    
    if fill_option == 'Drop':
        df = df.dropna(subset=numeric_columns)
    else:
        fill_value = (
            df[numeric_columns].mean() if fill_option == 'Mean' 
            else (df[numeric_columns].median() if fill_option == 'Median' 
                  else df[numeric_columns].mode().iloc[0])
        )
        df[numeric_columns] = df[numeric_columns].fillna(fill_value)

    st.dataframe(df)

    return df

def drop_column(df):
    st.subheader("5. Drop a column")
    columns_to_drop = st.multiselect("Select columns to drop:", df.columns)
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        st.dataframe(df)

    return df

def train_regression_model(df):
    st.subheader("7. Train a regression model")
    
    if df.empty:
        st.warning("Please upload a valid dataset.")
        return

    st.write("Select columns for X (features):")
    x_columns = st.multiselect("Select columns for X:", df.columns)

    if not x_columns:
        st.warning("Please select at least one column for X.")
        return

    st.write("Select the target column for Y:")
    y_column = st.selectbox("Select column for Y:", df.columns)

    if not y_column:
        st.warning("Please select a column for Y.")
        return

    df = df.dropna(subset=[y_column])

    X = df[x_columns]
    y = df[y_column]

    categorical_columns = X.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_columns)],
                                       remainder='passthrough')
        X = transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_option = st.selectbox("Choose a regression model:", ('Linear Regression', 'Random Forest'))

    if model_option == 'Linear Regression':
        model = LinearRegression()
    elif model_option == 'Random Forest':
        model = RandomForestRegressor()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write(f"Regression Model: {model_option}")
    st.write(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
    st.write(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}")
    st.write(f"Root Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred, squared=False)}")

    model_filename = f"{model_option}_model.joblib"
    joblib.dump(model, model_filename)
    st.success(f"Model saved as {model_filename}")

    st.subheader("8. Download the trained model")
    st.download_button(label="Download Model", data=open(model_filename, "rb").read(), file_name=model_filename)

def download_updated_dataset(df):
    st.subheader("6. Download the updated dataset")
    csv_file = df.to_csv(index=False)
    st.download_button("Download CSV", csv_file, "Updated_Dataset.csv", key="csv_download")

def main():
    st.title("CSV Dataset Analysis and Model Training App")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.info("File uploaded successfully!")
        df = load_data(uploaded_file)

        if not df.select_dtypes(include=['number']).empty:
            show_correlation(df)
            show_missing_values(df)
            df = handle_missing_values(df)

        df = drop_column(df)
        train_regression_model(df)
        download_updated_dataset(df)

if __name__ == "__main__":
    main()
