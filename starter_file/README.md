# Stroke Prediction Utility - Chad Puterbaugh

This project is meant to provide a utility to predict whether a patient with given demographic data is likely to develop a stroke. The data is sourced confidentially, but available via kaggle as linked in the Dataset section. This utility starts with a csv, and ends with a functional API endpoint where users can submit health information and receive a prediction on whether the demographic information represents someone likely to develop a stroke. 

## Project Set Up and Installation
This project requires no special instructions to set up. Simply initialize a compute, and run the automl.ipynb. Do not run the final cell if you intend to leave the API up and running. 

## Dataset

### Overview

Per Kaggle: 
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

The dataset has the following dimensions:
1. id: unique identifier
1. gender: "Male", "Female" or "Other"
1. age: age of the patient
1. hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
1. heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
1. ever_married: "No" or "Yes"
1. work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
1. Residence_type: "Rural" or "Urban"
1. avg_glucose_level: average glucose level in blood
1. bmi: body mass index
1. smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
1. stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

### Task
Initially, all 'N/A' remarks in columns are converted to `np.nan`'s, and all observations with NA's are blanket removed. All numeric columns are converted to numeric, the `id` column is removed, and the data is then saved in parquet format in the default datastore, and finally registered as a dataset `stroke-dataset`. 

For the `AutoML` job, this is as far as the data cleanup goes. `AutoML` performs featurization and achieves a very high accuracy. 

For the `hyperdrive` job, I perform a bit more featurization in `train.py`. `gender`, `ever_married`, and `Residence_type` are each converted into a binary integer (0 or 1) representation. Further, I perform one-hot encoding on `work_type` and `smoking_status`. 

### Access
The Kaggle data I used was accessed in May of 2022, and is provided as part of the package. 

## Automated ML
```
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 4, 
    "primary_metric" : 'balanced_accuracy'
}
```
`experiment_timeout_minutes` was selected to prevent any runaway experiments from being unfeasible economically.
`max_concurrent_iterations` was capped at 4 to match the number of nodes in the compute instance
`primary_metric` was set to `balanced_accuracy` to reflect that the a positive stroke prediction is much rarer in the dataset than the negative case.
See: https://neptune.ai/blog/balanced-accuracy

```
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="stroke",   
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```
The primary configurations worth noting in the `AutoMLConfig` are:
`task` this is a binary classification problem
`training_data` points to the dataset that was registered representing the cleaned stroke data
`label_column_name` as 'stroke' is the outcome variable we are trying to predict
`enable_early_stopping` allows any experiments to stop early to save compute costs
`featurization` set to `auto` allows AutoML to attempt to find the best set of features to represent the data

### Results

The AutoML model performed much better than my hyperdrive run with a balanced accuracy of {xyz}. The best performing model was an ensemble model consisting of:

*TODO* put in model parameters

The model could be improved as more data is collected. Similarly, more care to ensure that a balance of classes are represented in the data could yield better accuracy. The data itself could be better contextualized to represent when the observation actually developed the stroke as it is currently unclear. 

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
