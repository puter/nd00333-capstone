import pandas as pd
import numpy as np

def clean_data(data):
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().replace('N/A', np.nan).dropna()
    x_df.drop('id', inplace=True)
    x_df['bmi'] = x_df['bmi'].astype(float)
    x_df["gender"] = x_df.gender.apply(lambda s: 1 if s == "Male" else 0)
    x_df["ever_married"] = x_df.ever_married.apply(lambda s: 1 if s == "Yes" else 0)
    x_df["Residence_type"] = x_df.Residence_type.apply(lambda s: 1 if s == "Urban" else 0)
    work_type = pd.get_dummies(x_df.work_type, prefix="work_type")
    x_df.drop("work_type", inplace=True, axis=1)
    x_df = x_df.join(work_type)
    smoking_status = pd.get_dummies(x_df.smoking_status, prefix="smoking_status")
    x_df.drop("smoking_status", inplace=True, axis=1)
    x_df = x_df.join(smoking_status)

    return (x_df)