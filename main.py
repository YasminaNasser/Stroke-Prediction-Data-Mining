import pandas as pd
import Preprocessing  # Assuming preprocessing code is in Preprocessing.py
import numpy as np


# Load the dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(df)

# Preprocessing
df = Preprocessing.check_duplicate_rows(df)
df = Preprocessing.handle_categorical_missing_data(df)
df = Preprocessing.handle_numerical_missing_data_and_normalize(df)

# Identify numerical columns (including potential 'id' column)
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Detect outliers while excluding the 'id' column
outlier_indices = Preprocessing.detect_outliers(df, numerical_columns)

# Save outliers to a CSV file
outliers_df = df.loc[list(set(sum(outlier_indices.values(), [])))]
outliers_df.to_csv("outliers.csv", index=False)
print("\nOutliers exported to 'outliers.csv'.")

# Create a dataframe without outliers
outlier_rows = list(set(sum(outlier_indices.values(), [])))  # Flatten indices
df_without_outliers = df.drop(index=outlier_rows)

# Cap outliers in the original dataframe
df_capped_outliers = Preprocessing.cap_outliers(df, numerical_columns)

# Save results to CSV files
df_without_outliers.to_csv("data_without_outliers.csv", index=False)
print("\nDataset without outliers saved to 'data_without_outliers.csv'.")

df_capped_outliers.to_csv("data_capped_outliers.csv", index=False)
print("\nDataset with capped outliers saved to 'data_capped_outliers.csv'.")


