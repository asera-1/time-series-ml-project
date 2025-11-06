import os

import numpy as np
import pandas as pd
from create_hierarchical_data import assemble_predictions, save_hts_data
from scipy import linalg
from scipy.linalg import inv

from demandprediction.models.reconciliation_hts.reconciliation import To_Reconcile

column_labels = ["interval", "store_id", "category"]
date_columns = ["day", "month", "year"]
models = {
    "int_all_all":    {"colsample_bytree": 0.8, "learning_rate": 0.1,  "max_depth": 7,  "n_estimators": 1000, "subsample": 0.7}, 
    "int_all_total":  {"colsample_bytree": 0.9, "learning_rate": 0.05, "max_depth": 7,  "n_estimators": 800,  "subsample": 0.6},
    "int_total_total":  {"colsample_bytree": 0.7, "learning_rate": 0.01, "max_depth": 10, "n_estimators": 1000, "subsample": 0.6},
    # "hour_all_all":   {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 7,  "n_estimators": 1000, "subsample": 0.8},
    # "hour_all_total": {"colsample_bytree": 0.9, "learning_rate": 0.01, "max_depth": 10, "n_estimators": 1000, "subsample": 0.8},
    #"day_all_all":    {"colsample_bytree": 0.8, "learning_rate": 0.02, "max_depth": 8,  "n_estimators": 1000, "subsample": 0.7},
    # "day_all_total":  {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 7,  "n_estimators": 800,  "subsample": 0.6},
    "day_total_total":  {"colsample_bytree": 0.7, "learning_rate": 0.01, "max_depth": 10, "n_estimators": 1000, "subsample": 0.6},
    # "week_total_total": {"colsample_bytree": 0.8, "learning_rate": 0.01, "max_depth": 4,  "n_estimators": 500,  "subsample": 0.7}
}
output_dir = "category_model"
os.makedirs(output_dir, exist_ok=True)
idxs = save_hts_data(models, date_columns, output_dir)
print(idxs)
# load csvs
pred_csv_dirs = [f"{output_dir}/{model_name}.csv" for model_name in models.keys()]
actuals_csv_dirs = [f"{output_dir}/actuals_{model_name}.csv" for model_name in models.keys()]
error_df = pd.read_csv(f"{output_dir}/error_matrix.csv")
error_matrix = error_df.to_numpy()
print(error_matrix.shape)
complete_predictions = assemble_predictions(pred_csv_dirs, date_columns, column_labels)
complete_actuals = assemble_predictions(actuals_csv_dirs, date_columns, column_labels)

preds = complete_predictions.sort_values(by=column_labels, na_position='first')
actuals = complete_actuals.sort_values(by=column_labels, na_position='first')

preds_np = preds.to_numpy()
actuals_np = actuals.to_numpy()

preds_np = preds_np[:,len(column_labels):-1]
preds_np = preds_np[~idxs]
actuals_np = actuals_np[:,len(column_labels):-1]
actuals_np = actuals_np[~idxs]
complete_actuals = complete_actuals[~idxs]
complete_predictions = complete_predictions[~idxs]
complete_predictions.to_csv(f"{output_dir}/complete_predictions.csv", index=False)
print(preds_np.shape)
print(actuals_np.shape)
object = To_Reconcile(data = complete_actuals, columns_ordered=column_labels, base_forecasts= preds_np, real_values=actuals_np, error_matrix=error_matrix)
levels = object._get_indexes_level()

for method in ['OLS', 'BU', 'VS','MinTSh']:
    print(method)
    print(object.cross_score(metrics='rmse',reconcile_method=method, test_all=False, cv = 10))
    

os.makedirs("results", exist_ok=True)
summ_mat = object.summing_mat
lambd = object.lambd
summ_df = pd.DataFrame(summ_mat)
summ_df.to_csv(f"{output_dir}/summing_matrix.csv", index = False)
# write lambd to txt
with open(f"{output_dir}/lambd.txt", "w") as f:
    f.write(str(lambd))
