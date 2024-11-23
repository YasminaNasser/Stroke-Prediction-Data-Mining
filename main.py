import pandas as pd
import Preprocessing
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Example usage:
# Load the data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Check for and handle missing data in categorical features
has_nans, df = Preprocessing.handle_categorical_missing_data(df)
# print("------------------Final DataFrame------------------")
# print("Were there NaN values?:", has_nans)
# print(df)
