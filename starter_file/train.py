from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

#from azureml.core.workspace import Workspace
#ws = Workspace.from_config()


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

    y_df = x_df.pop("stroke")

    return (x_df, y_df)

run = Run.get_context()
ws = run.experiment.workspace
dataset = ws.datasets['stroke-dataset']

x, y = clean_data(dataset)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)



def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", float(args.C))
    print("Regularization Strength:", float(args.C))
    run.log("Max iterations:", int(args.max_iter))
    print("Max iterations:", int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    joblib.dump(value=model, filename='outputs/model.joblib')
    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    print("f1_score", f1_score(y_test, y_pred))
    run.log("Accuracy", float(accuracy))
    print("Accuracy", float(accuracy))
    os.makedirs('outputs', exist_ok=True)

if __name__ == '__main__':
    main()