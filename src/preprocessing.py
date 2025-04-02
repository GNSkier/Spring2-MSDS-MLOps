import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


predictor_raw = pd.read_csv(
    "/Users/skier/MSDS/Spring2/Spring2-MSDS-MLOps/labs/lab3/data/predictor_bronze.csv"
)
target_raw = pd.read_csv(
    "/Users/skier/MSDS/Spring2/Spring2-MSDS-MLOps/labs/lab3/data/target_bronze.csv"
)

predictor_raw_encoded = predictor_raw.copy()

# Track column transformations
column_mapping = {}
label_encoders = {}

# Find all object and category columns
string_columns = predictor_raw.select_dtypes(
    include=["object", "category"]
).columns

for col in string_columns:
    # For columns with many unique values, use label encoding
    le = LabelEncoder()
    predictor_raw_encoded[col + "_encoded"] = le.fit_transform(
        predictor_raw[col]
    )

    # Drop the original column
    predictor_raw_encoded = predictor_raw_encoded.drop(col, axis=1)

    # Store mapping information
    column_mapping[col] = [col + "_encoded"]
    label_encoders[col] = le

X_silver = predictor_raw_encoded.astype(float)
y_silver = target_raw

Data_Silver = pd.concat([X_silver, y_silver], axis=1)

Data_Silver.to_csv("labs/lab3/data/Iris_Silver.csv")
