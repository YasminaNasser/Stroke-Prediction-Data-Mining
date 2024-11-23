import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def handle_numerical_missing_data_and_normalize(ds):
    # Identify numerical features (excluding the last column)
    numerical_features = ds.iloc[:, :-1].select_dtypes(include=[np.number]).columns.tolist()
    print("------------------ Extracting Numerical Features ------------------")
    print(numerical_features)

    # Create a dataframe with only numerical features (excluding the target column)
    dataframe_N = ds[numerical_features]
    print("------------------ Numerical DataFrame ------------------------")
    print(dataframe_N)

    # Handle missing data: Replace NaN values with the mean of the column
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    dataframe_N_imputed = imp_mean.fit_transform(dataframe_N)

    # Convert the NumPy array back to a DataFrame with the original column names
    dataframe_N_imputed = pd.DataFrame(dataframe_N_imputed, columns=numerical_features)

    # Normalize the data to a range of [0, 1]
    normalizer = MinMaxScaler(feature_range=(0, 1))
    dataframe_N_normalized = normalizer.fit_transform(dataframe_N_imputed)

    # Convert the normalized array back to a DataFrame with the original column names
    dataframe_N_normalized = pd.DataFrame(dataframe_N_normalized, columns=numerical_features)

    # Update the original dataset with the normalized numerical features
    ds[numerical_features] = dataframe_N_normalized

    print("------------------- Data After Handling Missing Values and Normalization ---------------")
    print(ds)
    return ds



def handle_categorical_missing_data(df):
    # Extract categorical features
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    print("------------------Extracting Categorical Features------------------")
    # print("Categorical Features:", categorical_features)

    # Check if there are NaN values in categorical features
    has_nans = df[categorical_features].isnull().any().any()
    print("Are there NaN values in categorical features?:", has_nans)

    # Handle missing values if NaNs exist
    if has_nans:
        imp_freq = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        df[categorical_features] = imp_freq.fit_transform(df[categorical_features])
        print("Categorical features after handling missing data (most frequent):")
        print(df[categorical_features])

    return has_nans, df


