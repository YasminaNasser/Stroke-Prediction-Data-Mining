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

def detect_outliers(df, columns):
    # Exclude the 'id' column if it exists
    if 'id' in columns:
        columns.remove('id')

    outlier_indices = {}

    print("------------------Detecting Outliers------------------")
    for column in columns:
        Q1 = df[column].quantile(0.25)  # First quartile
        Q3 = df[column].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1                   # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        if not outliers.empty:
            print(f"Outliers in column '{column}':")
            print(outliers)
            outlier_indices[column] = outliers.index.tolist()
        else:
            print(f"No outliers detected in column '{column}'.")

    return outlier_indices

def cap_outliers(df, columns):
    capped_df = df.copy()
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap values outside the bounds
        capped_df[column] = np.where(capped_df[column] < lower_bound, lower_bound,
                                     np.where(capped_df[column] > upper_bound, upper_bound, capped_df[column]))
    return capped_df



