import itertools
import logging
import os
import pprint
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
import yaml
from eval_run import train_and_eval
from prepare_data import DataConfig, preprocess_data
from prophet.make_holidays import make_holidays_df

logger = logging.getLogger("mlflow")
print(logging.getLogger().handlers)

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    search_type : str = "grid"
    model_type : str = "xgb"
    model_params: dict = field(default_factory=lambda: {'max_depth': [6], 'learning_rate': [0.1], 'n_estimators': [600]})
    prediction_type: list[str] = field(default_factory=lambda: ["single", "multi"])
    add_missing_days: list[bool] = field(default_factory=lambda: [True, False])
    add_missing_intervals: list[bool] = field(default_factory=lambda: [True, False])
    store_ids: list[list[int]] = field(default_factory=lambda: [[1], [2]])
    temporal_level: str = "interval"
    n_features: int = 5 
    features: list[str] = field(default_factory=lambda: ["year", "month", "week", "day", "weekday", "hour", "interval", "open", "holiday", "semester_phase", "semester_period"])
    targets: list[str] = field(default_factory=lambda: ["MAIN", "SIDE", "SOUP", "DESSERT", "SALAD", "BOTTLE", "BAKED_GOOD", "CONDIMENT", "OTHER"])
    n_runs: int = 5

@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    
    add_missing_days: bool = True
    add_missing_intervals: bool = True
    store_ids: list[int] = field(default_factory=lambda: [1, 2])
    temporal_level: str = "interval"
    features: list[str] = field(default_factory=lambda: [])
    targets: list[str] = field(default_factory=lambda: ["TOTAL_SALES"])

    model_type: str = "xgb" # "prophet"
    prediction_type: str = "single" # "multi"
    model_params: dict = field(default_factory=lambda: {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 600})
    
    def __str__(self):
        return pprint.pformat(asdict(self))
    
def check_and_delete_experiment(name: str)->bool:
    experiment = mlflow.get_experiment_by_name(name)

    if experiment:
        logger.info(f"Experiment '{name}' already exists with ID {experiment.experiment_id}")

        # Prompt user for confirmation
        user_input = input(f"Do you want to overwrite experiment '{name}' (yes/no)? ").strip().lower()

        if user_input == 'yes':
            mlflow.delete_experiment(experiment.experiment_id)
            logger.info(f"Experiment '{name}' marked as deleted.")
            return True
        else:
            logger.info(f"Experiment '{name}' was not deleted.")
            return False
    else:
        logger.info(f"Experiment '{name}' does not exist yet.")
        return True



def load_data(config: RunConfig) -> Tuple[pd.DataFrame, list[str], dict]:
    time_features = list(set(config.features).intersection({"year", "month", "week", "day", "weekday", "hour", "interval"}))
    calendar_features = list(set(config.features).intersection({"open", "holiday", "semester_phase", "semester_period"}))
    model_params = config.model_params 
    if config.model_type == "prophet" and "holiday" in calendar_features:
        calendar_features.remove("holiday")
        model_params["holidays"] = make_holidays_df(year_list=[2023, 2024, 2025], country='DE', state='NI')
    if "semester_phase" in calendar_features:
        calendar_features.remove("semester_phase")
        calendar_features += ["semester_phase_uni","semester_phase_hs"]
    if "semester_period" in calendar_features:
        calendar_features.remove("semester_period") 
        calendar_features += ["semester_uni","semester_hs"]
    
    features = time_features + calendar_features
    if config.store_ids:
        if len(config.store_ids) > 1:
            features += ["store_id"]

    data_config = DataConfig(
        temporal_level=config.temporal_level,
        time_columns=list(time_features),
        calendar_columns=list(calendar_features),
        category_columns=config.targets,
        add_missing_intervals=config.add_missing_intervals,
        add_missing_days=config.add_missing_days,
        store_ids=config.store_ids
    )

    df = preprocess_data(data_config)
    logger.info(df["year"].unique())  
    return df, features, model_params

def excecute_run(config: RunConfig=None, idx :int = 0, prefix :str = ""):
    if config is None:
        run = sys.argv[1:][0]
        with open(run) as f:
            config = RunConfig(**yaml.safe_load(f))
    
        mlflow.set_tracking_uri="https://mlflow-aip-2.visiolab.io" 
        mlflow.set_experiment("test")
        logger.info(f"set up tracking at {mlflow.get_tracking_uri()}")
    df, features, model_params = load_data(config)
    logger.info("Starting run")

    with mlflow.start_run(run_name=f"run{idx}", nested=True):
        mlflow.log_params(asdict(config))
        mlflow.log_params(model_params)
        prefix += f"/{mlflow.active_run().info.run_id}"
        if config.prediction_type == "multi":
            logger.info("training model to do multi category prediction")
            train_and_eval(df, config.model_type, model_params, features, config.targets, config.seed, prefix)
        else:
            logger.info("training models to do single category prediction")
            for category in config.targets:
                logger.info(f"training model for {category}")
                train_and_eval(df, config.model_type, model_params, features, [category], config.seed, f"{prefix}/{category}")

    logger.info(f"Run {idx} completed")

def grid_search(config: ExperimentConfig) -> list[RunConfig]:
    n = config.n_features
    if n < 0:
        features = [config.features]
    else:
        features = []
        for comb in itertools.combinations(config.features, n):
            features.append(list(comb))
        
    data_params = {
        "prediction_type": config.prediction_type,
        "add_missing_intervals": config.add_missing_intervals,
        "add_missing_days": config.add_missing_days,
        "store_ids": config.store_ids,
        "features": features,
        "targets": [config.targets],
        "temporal_level": [config.temporal_level]
    }

    model_params = {**config.model_params}
    keys, values = zip(*model_params.items())
    model_params_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    params = {**data_params, "model_params": model_params_grid}
    
    keys, values = zip(*params.items())
    combos = [RunConfig(**dict(zip(keys, v))) for v in itertools.product(*values)]
    return combos
    
def random_search(config: ExperimentConfig) -> list[RunConfig]:
    runs = []
    for i in range(config.n_runs):
        run = {}
        run["seed"] = config.seed
        run["features"] = config.features if config.n_features < 0 else np.random.choice(config.features, size=config.n_features, replace=False)
        run["prediction_type"] = np.random.choice(config.prediction_type)
        run["add_missing_intervals"] = np.random.choice(config.add_missing_intervals)
        run["add_missing_days"] = np.random.choice(config.add_missing_days)
        
        store_idx = np.random.choice(np.arange(len(config.store_ids)))
        run["store_ids"] = None if config.store_ids == "None" else config.store_ids[store_idx]
        run["targets"] = config.targets
        run["temporal_level"] = config.temporal_level
        run["model_params"] = {}
        for param, value in config.model_params.items():
            run["model_params"][param] = np.random.choice(value)

        runs.append(RunConfig(**run))

    return runs
    
def conduct_experiment():
    experiment_path = sys.argv[1:][0]
    with open(experiment_path) as f:
        experiment_raw = yaml.safe_load(f)
        config_raw = experiment_raw["experiment"]
        meta_raw = experiment_raw["meta"]
        config = ExperimentConfig(**config_raw)

    mlflow.set_tracking_uri="https://mlflow-aip-2.visiolab.io"
    logger.info(f"set up tracking at {mlflow.get_tracking_uri()}")

    name = meta_raw["name"]
    tags = meta_raw["tags"]

    if check_and_delete_experiment(name): 
        mlflow.create_experiment(name=name, tags=tags)

    mlflow.set_experiment(name)
    logger.info(f"Starting experiment {name}")
    if config.search_type == "grid":
        runs = grid_search(config)
    else:
        runs = random_search(config)

    for idx, run in enumerate(runs):
        excecute_run(run, idx, name)    
    logger.info(f"Experiment {name} completed")

if __name__ == "__main__":
    conduct_experiment()
    
