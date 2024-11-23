import numpy as np
from sklearn.impute import SimpleImputer

def handle_categorical_missing_data(df):
    # Extract categorical features
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    print("------------------Extracting Categorical Features------------------")
    print("Categorical Features:", categorical_features)

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
