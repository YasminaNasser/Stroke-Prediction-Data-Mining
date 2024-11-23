import pandas as pd
import Preprocessing
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(df)
# Check for and handle missing data in categorical features
df=Preprocessing.check_duplicate_rows(df)
df = Preprocessing.handle_categorical_missing_data(df)
df=Preprocessing.handle_numerical_missing_data_and_normalize(df)

