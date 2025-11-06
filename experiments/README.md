## Experiment pipeline
In this workspace we can define and run experiments for XGBoost and Prophet models and track evals in mlflow.

Navigate to experiments directory and excecute:
``` bash
uv run main.py "path/to/your/config"
```

## Setup
#### MLFlow
To track experiments on the remote mlflow server (https://mlflow-aip-2.visiolab.io) we need to setup authentification, using the username and password, in the terminal run:
``` bash
export MLFLOW_TRACKING_USERNAME=username
export MLFLOW_TRACKING_PASSWORD=password
export MLFLOW_TRACKING_URI=https://mlflow-aip-2.visiolab.io
```
#### Experiment Config
In the configs folder, define the experiment. Under the "meta" key, specify an experiment name and tags, which can include a description and hypothesis about the experiment.
Under the "experiment" key, specify the hyperparameter-ranges that the individual run configs will be constructed from.

## Model Evaluation
The models folder includes training and evaluation procedures for both Prophet and XGBoost.

#### XGBoost
The train_xgb.py file specifies a train_and_eval function which takes as an input a range of run- and model-parameters and then excecutes repeated K-fold sampling to test model performance across a variety of train/test splits. In addition an identical model is trained on a time-based split and a plot of predictions (up to 2025-01-31) as well as the model itself, are saved in a local folder (artifacts/{run_id}/).

#### Prophet


