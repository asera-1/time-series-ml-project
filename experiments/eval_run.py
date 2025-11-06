import logging
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, TimeSeriesSplit

from demandprediction.models.prophet_model import ProphetTrainer
from demandprediction.models.xgb_model import XGBTrainer

logger = logging.getLogger("mlflow")

def split_test_train(df):
    df_kfold = df[df["date"] <= "2024-12-31"]
    df_train, df_test = df[df["date"] < "2024-09-01"], df[df["date"] >= "2024-09-01"]
    return df_kfold, df_train, df_test

def plot_pred(dates, y, y_pred, target, dir):
    height = len(target)
    fig, ax = plt.subplots(height, constrained_layout=True, figsize=(12, 4 * height))
    # make sure ax has at least 1 dimenstion
    ax = np.atleast_1d(ax)
    ax = ax.reshape(height)

    for i in range(height):
        if height > 1:
            current_y, current_y_pred = y.iloc[:, i], y_pred[:, i]
            current_target = target[i]
        else:
            current_y, current_y_pred = y, y_pred
            current_target = target
        ax[i].plot(dates, current_y, label="actual", alpha=0.5, linewidth=3)
        ax[i].plot(dates, current_y_pred, label="predicted")
        ax[i].set_title(f"Predictions for {current_target}", fontsize=16)
        ax[i].legend()
        ax[i].set_ylabel("Sales")
        ax[i].set_xlabel("Date")
    
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    fig.savefig(dir)
    plt.close(fig)


def kfold_training(df, model, cv, mlflow_prefix="") -> dict:
    metrics_list = defaultdict(list)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df)):
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
        model.fit(df_train)   

        if "store_id" in df_test.columns:
            for store_id in df_test["store_id"].unique():
                df_test_store = df_test[df_test["store_id"] == store_id].copy()
                metrics = model.eval(df_test_store)
                for metric, value in metrics.items():
                    # mlflow.log_metric(f"{metric}_{store_id}", value, step=fold_idx)
                    metrics_list[f"{metric}_{store_id}"].append(value)
        else:
            metrics = model.eval(df_test)
            for metric, value in metrics.items():
                # mlflow.log_metric(metric, value, step=fold_idx)
                metrics_list[metric].append(value)
    
    total_metric = {}
    # Aggregate metrics
    for metric, values in metrics_list.items():
        mlflow.log_metric(f"{metric}_mean", float(np.mean(values)))
        total_metric[f"{metric}_mean"] = float(np.mean(values))
        mlflow.log_metric(f"{metric}_std", float(np.std(values)))
        total_metric[f"{metric}_std"] = float(np.std(values))

    return total_metric


def time_split_training(df_train, df_test, model, target, mlflow_prefix=""):
    dir = f"artifacts/{mlflow_prefix}"
    os.makedirs(os.path.dirname(dir), exist_ok=True)

    model.fit(df_train)
    y_pred = model.predict(df_test)    
    if "store_id" in df_test.columns:
        for store_id in df_test["store_id"].unique():
            df_test_store = df_test[df_test["store_id"] == store_id].copy()
            dates = df_test_store["date"]
            y_store = df_test_store[target]
            y_pred_store = model.predict(df_test_store)
            plot_pred(dates, y_store, y_pred_store, target, f"{dir}/model_pred_store_{store_id}.png")
    else:
        plot_pred(df_test["date"], df_test[target], y_pred, target, f"{dir}/model_pred.png")

    model.save_model(dir)

def train_and_eval(df, model_type, model_params, features, target, seed, mlflow_prefix) -> dict:

    df_kfold, df_train, df_test = split_test_train(df)
    
    if model_type == "xgb":
        model = XGBTrainer(model_params, features, target)
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
    else:
        model = ProphetTrainer(model_params, features, target)
        cv = TimeSeriesSplit(n_splits=5)

    time_split_training(df_train, df_test, model, target, mlflow_prefix)
    kfold_training(df_kfold, model, cv, mlflow_prefix)
    
    

    