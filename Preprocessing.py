import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def check_duplicate_rows(df, remove_duplicates=False):
    # Check for duplicate rows
    duplicate_rows = df[df.duplicated()]

    print("------------------Checking for Duplicate Rows------------------")
    if not duplicate_rows.empty:
        print(f"Number of duplicate rows: {len(duplicate_rows)}")
        print("Duplicate Rows:\n", duplicate_rows)
    else:
        print("No duplicate rows found.")

    # Remove duplicates if specified
    if remove_duplicates:
        df = df.drop_duplicates()
        print("\nDuplicate rows removed.")

    return df
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
    print("Categorical Features:", categorical_features)

    total_rows = len(df)
    threshold = total_rows* 0.01  # 1%

    for feature in categorical_features:
        print(f"\nProcessing Feature: {feature}")

        # Get value counts for the feature
        value_counts = df[feature].value_counts(dropna=False)
        print(f"Value Counts:\n{value_counts}")

        # Identify frequent and infrequent values
        frequent_values = value_counts[value_counts > threshold].index
        infrequent_values = value_counts[value_counts <= threshold].index
        print(f"Frequent Values (Threshold > {threshold:.2f}): {list(frequent_values)}")
        print(f"Infrequent Values (Threshold <= {threshold:.2f}): {list(infrequent_values)}")

        # Replace infrequent values and NaNs with the most frequent value
        most_frequent_value = value_counts.idxmax()
        df[feature] = df[feature].apply(
            lambda x: x if x in frequent_values else most_frequent_value
        )

        print(f"Updated Feature Values:\n{df[feature].value_counts(dropna=False)}")
        print("-------------------------------------------------------------------")


    return df




