import os

import numpy as np
import pandas as pd
from prepare_data import DataConfig, preprocess_data

from demandprediction.models.xgb_model import XGBTrainer

TIME_FEATURES = ["year", "month", "week", "weekday","day", "hour", "interval"]

CALENDAR_FEATURES = {
    "interval": ["semester_phase_uni", "semester_phase_hs", "semester_uni", "semester_hs", "holiday", "open"],
    "hour": ["semester_phase_uni", "semester_phase_hs", "semester_uni", "semester_hs", "holiday", "open"],
    "day": ["semester_phase_uni", "semester_phase_hs", "semester_uni", "semester_hs", "holiday"],
    "week": ["semester_phase_uni", "semester_phase_hs", "semester_uni", "semester_hs"]
}
TARGETS = ["MAIN", "SIDE", "SOUP", "DESSERT", "SALAD", "BOTTLE", "BAKED_GOOD", "CONDIMENT", "OTHER"]

# FINAL_COL = {
#     "interval": ["weekday", "hour","interval"],
#     "hour": ["weekday", "hour"],
#     "day": ["weekday"],
#     "week": []
# }

FINAL_COL = {
    "interval": ["interval"],
    "hour": ["hour"],
    "day": []
}


def split_df(df, train_dates):
    train_mask = df["date"].isin(train_dates)
    return df[train_mask], df[~train_mask]

def process_predictions(predictions: np.ndarray) -> np.ndarray:
    # replace x < 0 with 0
    predictions[predictions < 0] = 0
    # round
    predictions = predictions.round()
    return predictions

def predict_2025(model_name, model_params, output_dir="results"):
    model_list = model_name.split("_")
    print(f"predict 2025 for {model_name}")
    if model_list[0] == "int":
        model_list[0] = "interval"
    temporal_level = model_list[0]
    idx = TIME_FEATURES.index(model_list[0])
    time_features = TIME_FEATURES[:idx+1]
    print(time_features)
    calendar_features = CALENDAR_FEATURES[model_list[0]]
    store_ids = [1, 2] if model_list[1] == "all" else None
    targets = ["TOTAL_SALES"] if model_list[-1] == "total" else TARGETS

    data_config = DataConfig(
        temporal_level=temporal_level,
        time_columns=time_features,
        calendar_columns=calendar_features,
        category_columns=targets,
        add_missing_intervals=True,
        add_missing_days=True,
        store_ids=store_ids,
        end_date="2025-12-31"
    )
    df = preprocess_data(data_config)
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[df["date"] < "2025-01-01"]
    test_df = df[df["date"] >= "2025-01-01"]
    model = XGBTrainer(model_params, time_features+calendar_features, targets)
    model.fit(train_df)
    predictions = model.predict(test_df)
    predictions = process_predictions(predictions)
    predictions = pd.DataFrame(predictions, columns=targets)
    predictions["date"] = test_df["date"].values
    if store_ids is not None:
        predictions["store_id"] = test_df["store_id"].values
    predictions[time_features] = test_df[time_features].values
    predictions.to_csv(f"{output_dir}/{model_name}.csv", index=False)
    
def save_actuals(model_name, output_dir):
    model_list = model_name.split("_")
    if model_list[0] == "int":
        model_list[0] = "interval"
    temporal_level = model_list[0]
    idx = TIME_FEATURES.index(model_list[0])
    time_features = TIME_FEATURES[:idx+1]
    calendar_features = CALENDAR_FEATURES[model_list[0]]
    store_ids = [1, 2] if model_list[1] == "all" else None
    targets = ["TOTAL_SALES"] if model_list[-1] == "total" else TARGETS
    data_config = DataConfig(
        temporal_level=temporal_level,
        time_columns=time_features,
        calendar_columns=calendar_features,
        category_columns=targets,
        add_missing_intervals=True,
        add_missing_days=True,
        store_ids=store_ids,
        end_date="2025-12-31"
    )
    df = preprocess_data(data_config)
    df["date"] = pd.to_datetime(df["date"])
    test_df = df[df["date"] >= "2025-01-01"]
    if model_list[1] == "all":
        test_df = test_df[["store_id", "date"]+targets+time_features]
    else:
        test_df = test_df[targets+time_features+["date"]]
    test_df.to_csv(f"{output_dir}/actuals_{model_name}.csv", index=False)

def assemble_predictions(prediction_csv_dirs, date_columns, column_labels):
    """
    Assemble predictions from multiple models
    csv files with predictions need to have the format:
    data, (any or all of) hour, interval, store_id, target categories
    """
    complete_predictions = pd.DataFrame()
    for dir in prediction_csv_dirs:
        print(dir)
        predictions = pd.read_csv(dir) 
        predictions["date"] = predictions[date_columns].apply(lambda x: "-".join(x.astype(str)), axis=1)
        cols = list(predictions.columns)
        pred_cols = list(set(cols).intersection(set(TARGETS+["TOTAL_SALES"])))
        cols = list(set(cols).intersection(set(column_labels)))
        predictions = predictions[["date"]+cols+pred_cols]
        if "interval" in cols and "hour" in cols:
            print("interval_adjustment")
            predictions["interval"] = predictions["interval"].apply(lambda x: x%4)
        if "TOTAL_SALES" in cols:
            predictions["prediction"] = predictions["TOTAL_SALES"]
        else:
            predictions = predictions.melt(
                id_vars=cols+["date"], var_name="category", value_name="prediction"
            )
            cols += ["category"]
        
        predictions_pivot = predictions.pivot_table(
            index=cols, columns="date", values="prediction", aggfunc="first"
        ).reset_index()
        complete_predictions = pd.concat([complete_predictions, predictions_pivot], ignore_index=True)
    return complete_predictions 

    
def create_error_matrix(actuals, preds, column_labels):
    # actuals.drop("index", inplace=True, axis=1)
    actuals = actuals.sort_values(by=column_labels, na_position='first')
    actuals_np_error = actuals.to_numpy()
    # preds.drop("index", inplace=True, axis=1)
    preds = preds.sort_values(by=column_labels, na_position='first')
    preds_np_error = preds.to_numpy()
    actuals_np_error = actuals_np_error[:,len(column_labels):-1]
    preds_np_error = preds_np_error[:,len(column_labels):-1]
    error_matrix = (preds_np_error - actuals_np_error).astype(np.float64)
    idxs = np.all(error_matrix == 0, axis=1)
    error_matrix = error_matrix[~np.all(error_matrix == 0, axis=1)]
    # transpose error matrix
    error_matrix = error_matrix
    return error_matrix, idxs

def compute_residuals(models: dict, date_columns, split_ratio: float = 0.5, output_dir: str = "results"):
    column_labels = set()
    complete_predictions = pd.DataFrame()
    complete_actuals = pd.DataFrame()
    train_dates = None
    for id, model_str in enumerate(models.keys()):
        model_list = model_str.split("_")
        print(model_str)
        if model_list[0] == "int":
            model_list[0] = "interval"
        temporal_level = model_list[0]
        idx = TIME_FEATURES.index(model_list[0])
        time_features = TIME_FEATURES[:idx+1]
        calendar_features = CALENDAR_FEATURES[model_list[0]]
        store_ids = [1, 2] if model_list[1] == "all" else None
        targets = ["TOTAL_SALES"] if model_list[-1] == "total" else TARGETS

        data_config = DataConfig(
            temporal_level=temporal_level,
            time_columns=time_features,
            calendar_columns=calendar_features,
            category_columns=targets,
            add_missing_intervals=True,
            add_missing_days=True,
            store_ids=store_ids,
            end_date="2024-12-31"
        )

        df = preprocess_data(data_config)
        df["date"] = df[date_columns].apply(lambda x: "-".join(x.astype(str)), axis=1)
        if train_dates is None:
            dates = df["date"].unique()
            train_dates = np.random.choice(dates, size=int(len(dates)*split_ratio), replace=False)
        model = XGBTrainer(models[model_str], time_features + calendar_features, targets)
        cols = FINAL_COL[model_list[0]].copy()
        train_df, test_df = split_df(df, train_dates)
        model.fit(train_df)
        predictions = model.predict(test_df)
        predictions = process_predictions(predictions)
        predictions = pd.DataFrame(predictions, columns=targets)
        if model_list[1] == "all":
            cols += ["store_id"]
        predictions[cols + ["date"]] = test_df[cols + ["date"]].values
        predictions = predictions[["date"] + cols + targets]
        actuals = test_df[["date"] + cols + targets]
        print(cols)
        if "interval" in cols and "hour" in cols:
            print("interval_adjustment")
            predictions["interval"] = predictions["interval"].apply(lambda x: x%4)
            actuals["interval"] = actuals["interval"].apply(lambda x: x%4)
        if targets != ["TOTAL_SALES"]:
            predictions = predictions.melt(
                id_vars=cols + ["date"], var_name="category", value_name="prediction"
            )
            actuals = actuals.melt(
                id_vars=cols + ["date"], var_name="category", value_name="actual"
            )
            cols += ["category"]
        else:
            predictions["prediction"] = predictions["TOTAL_SALES"]
            actuals["actual"] = actuals["TOTAL_SALES"]

        predictions = predictions[["date"] + cols + ["prediction"]]
        actuals = actuals[["date"] + cols + ["actual"]]

        predictions_pivot = predictions.pivot_table(
            index=cols, columns="date", values="prediction", aggfunc="first"
        ).reset_index()
        actuals_pivot = actuals.pivot_table(
            index=cols, columns="date", values="actual", aggfunc="first"
        ).reset_index()

        complete_predictions = pd.concat([complete_predictions, predictions_pivot], ignore_index=True)
        complete_actuals = pd.concat([complete_actuals, actuals_pivot], ignore_index=True)
        column_labels = column_labels.union(set(cols))

    error_matrix, idxs = create_error_matrix(complete_predictions, complete_actuals, list(column_labels))
    error_df = pd.DataFrame(error_matrix)
    error_df.to_csv(f"{output_dir}/error_matrix.csv", index=False)
    return idxs

def save_hts_data(models, date_columns, output_dir):
    idxs = compute_residuals(models, date_columns, split_ratio=0.5, output_dir=output_dir)

    for model_name, model_params in models.items():
        predict_2025(model_name, model_params, output_dir=output_dir)
        save_actuals(model_name, output_dir=output_dir)
        pass
    return idxs