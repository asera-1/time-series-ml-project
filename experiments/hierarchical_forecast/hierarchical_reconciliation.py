import os

import numpy as np
import pandas as pd
from create_hierarchical_data import assemble_predictions, save_hts_data
from scipy import linalg
from scipy.linalg import inv

from demandprediction.models.reconciliation_hts.reconciliation import To_Reconcile


def temporal_reconciliation(features):
    date = features["day"] + "-" + features["month"] + "-" + features["year"]
    column_labels = ["hour", "interval", "store_id", "category"]
    dir = "temporal_model"
    error_df = pd.read_csv(f"{dir}/error_matrix.csv")
    error_matrix = error_df.to_numpy()
    predictions = pd.read_csv(f"{dir}/complete_predictions.csv")
    predictions = predictions[predictions["date"] == date]
    preds_np = predictions[date].to_numpy()
    preds_np = preds_np[:,len(column_labels):-1]

    summing_mat = pd.read_csv(f"{dir}/summing_matrix.csv")
    summing_mat = summing_mat.to_numpy().astype(float)  # Ensure all elements are floats
    with open(f"{dir}/lambd.txt", "r") as f:
        lambd = float(f.read())
    object = To_Reconcile(base_forecasts= preds_np, error_matrix=error_matrix, summing_mat=summing_mat, lambd=lambd)
    predictions["reconciled_preds"] = pd.Series(object.reconcile(method='MinTSh', reconcile_all=True, show_lambda=True))

    predictions[predictions["interval"]==features["interval"] & predictions["hour"]==features["hour"] & predictions["store_id"]==features["store_id"]]
    return predictions["reconciled_preds"]

