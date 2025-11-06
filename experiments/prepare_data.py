import os
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from demandprediction.preprocessing import calendar_features

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

@dataclass(frozen=True)
class DataConfig:
    temporal_level: str = "day" # "interval"
    time_columns: list[str] = field(
        default_factory=lambda: ["year", "month", "week", "day", "hour", "interval"]
    )
    calendar_columns: list[str] = field(
        default_factory=lambda: ["holiday", "open", "semester_uni", "semester_hs", "semester_phase_uni", "semester_phase_hs"]
    )
    category_columns: list[str] = field(
        default_factory=lambda: ["MAIN","SIDE","SOUP","DESSERT","SALAD","BOTTLE","BAKED_GOOD","CONDIMENT","OTHER"]
    )
    cap_outliers: bool = False 
    store_ids: list[int] = field(default_factory=lambda: [1, 2])
    add_missing_days: bool = True
    add_missing_intervals: bool = True
    add_missing_data: bool = False
    start_date: str = "2023-01-01"
    end_date: str = "2025-01-31"

def load_data_combined(adjusted_data: bool = True) -> pd.DataFrame:
    """
    load and sum up values for store 1 and store 2
    """
    if adjusted_data:
        file_prefix = "cleaned_adjusted"
    else:
        file_prefix = "cleaned"
    store1 = pd.read_csv(f"{DATA_DIR}/{file_prefix}_store1.csv")
    store2 = pd.read_csv(f"{DATA_DIR}/{file_prefix}_store2.csv")
    # add up values for store 1 and store 1 across the same "datetime"
    concat_df = pd.concat([store1, store2])
    summed_df = concat_df.groupby('datetime', as_index=False).sum()
    return summed_df

def load_data_single(store: int, adjusted_data: bool = True) -> pd.DataFrame:
    """
    load the default dataframe for the indicated store
    """
    if adjusted_data:
        file_prefix = "cleaned_adjusted"
    else:
        file_prefix = "cleaned"
    store_df = pd.read_csv(f"{DATA_DIR}/{file_prefix}_store{store}.csv")
    store_df["store_id"] = store
    return store_df

def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["week"] = df["datetime"].dt.isocalendar().week
    df["day"] = df["datetime"].dt.day
    df["weekday"] = df["datetime"].dt.weekday
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["interval"] = df["hour"] * 4 + df["minute"] // 15
    return df

def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["holiday"] = calendar_features.assign_holiday(df["datetime"])
    df["open"] = calendar_features.assign_open(df["datetime"])
    df["semester_uni"] = calendar_features.uni_semester_period(df["datetime"])
    df["semester_hs"] = calendar_features.hs_semester_period(df["datetime"])
    df["semester_phase_uni"] = calendar_features.uni_semester_phase(df["datetime"])
    df["semester_phase_hs"] = calendar_features.hs_semester_phase(df["datetime"])
    return df

def include_timestamps(df: pd.DataFrame, type: str, start: str="2023-01-01", end: str="2024-12-31") -> pd.DataFrame:
    """
    input: df with datetime column, type of timestamps to add, range of timestamps
    output: df with updated timestamps
    """
    df["time"] = pd.to_datetime(df["datetime"]).dt.time
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    time_range = df["time"].unique()
    date_range = df["date"].unique()
    if type == "interval":
        time_range = pd.date_range("00:00", "23:45", freq="15min").time
    elif type == "day":
        date_range = pd.date_range(start, end)
    elif type == "period":
        max_date = df["date"].max()
        end_date = pd.to_datetime(end).date()
        min_date = df["date"].min()
        start_date = pd.to_datetime(start).date()
        date_range = pd.DatetimeIndex(date_range)
        date_range = date_range[(date_range >= pd.Timestamp(start_date)) & (date_range <= pd.Timestamp(end_date))]
        if max_date < end_date:
            date_append = pd.date_range(start=max_date + timedelta(days=1), end=end_date)
            date_range = date_range.union(date_append)
        if min_date > start_date:
            date_prepend = pd.date_range(start=start_date, end=min_date - timedelta(days=1))
            date_range = date_prepend.union(date_range)
    else:
        return df

    if "store_id" in df.columns:
        store_ids = df["store_id"].unique()
        full_index = pd.MultiIndex.from_product([date_range, time_range, store_ids], names=["date", "time", "store_id"])
        df_full = df.set_index(["date", "time", "store_id"]).reindex(full_index).fillna(0).reset_index()
    else:
        full_index = pd.MultiIndex.from_product([date_range, time_range], names=["date", "time"])
        df_full = df.set_index(["date", "time"]).reindex(full_index).fillna(0).reset_index()

    df_full["datetime"] = pd.to_datetime(
        df_full["date"].astype(str) + " " + df_full["time"].astype(str)
    )
    df_full.sort_values(by="datetime", inplace=True)
    return df_full


def min_date_column(df: pd.DataFrame, temporal_level: str)-> pd.DataFrame:
    """
    Construct a date column based on min_level
    """
    df["datetime"] = pd.to_datetime(df["datetime"])
    date_level = {"interval": "day","hour": "day","day": "day","week": "month","month": "month","semester": "year","year": "year"}
    min_date_level = date_level[temporal_level]

    if min_date_level == "month":
        df["date"] = df["datetime"].dt.strftime("%Y-%m") 
    elif min_date_level == "year":
        df["date"] = df["datetime"].dt.strftime("%Y")
    return df

def robust_outlier_cap(column: pd.Series, lower_percentile=1, upper_percentile=99):
    """ Winsorizes extreme values based on percentiles to avoid hard clipping. """
    lower_bound, upper_bound = np.percentile(column, [lower_percentile, upper_percentile])
    return np.clip(column, lower_bound, upper_bound)

def preprocess_data(config: DataConfig) -> pd.DataFrame:
    """
    Load clean dataframe, preprocess and add features based on config.
    """
    if config.store_ids is None:
        df = load_data_combined()
        cols = ["date"] + config.time_columns + config.calendar_columns   
    else:
        dfs = []
        for store_id in config.store_ids:
            dfs.append(load_data_single(store_id))
        df = pd.concat(dfs, axis=0)
        cols = ["date"] + ["store_id"] + config.time_columns + config.calendar_columns 

    df["datetime"] = pd.to_datetime(df["datetime"])

    if config.add_missing_intervals:
        df = include_timestamps(df, type="interval")
    if config.add_missing_days:
        df = include_timestamps(df, type="day", start=config.start_date, end=config.end_date)
    else:
        df = include_timestamps(df, type="period", start=config.start_date, end=config.end_date)

    df = add_time_columns(df)
    df = add_calendar_columns(df)

    df = min_date_column(df, config.temporal_level)
    group_cols = list(set(cols + [config.temporal_level]))
    output_df = df.groupby(group_cols)[config.category_columns].sum().reset_index()
    output_df = output_df[cols + config.category_columns]
    output_df.sort_values(by="date", inplace=True)
    if config.cap_outliers:
        for column in config.category_columns:
            output_df[column] = robust_outlier_cap(output_df[column])

    return output_df 


# config = DataConfig(
#     temporal_level="week",
#     time_columns=["month", "week"],
#     calendar_columns=["holiday", "open", "semester_uni"],
#     category_columns=["TOTAL_SALES"],
#     add_missing_intervals=False,
#     add_missing_days=False,
#     store_ids=[1,2],
#     end_date="2024-04-10",
#     cap_outliers=True,
# )
# df = preprocess_data(config)
# print(df.head(20))
# print(df.tail(20))
# 
# df_pivoted = df.pivot(columns="store_id", values="TOTAL_SALES")
# df_pivoted.columns = [f"TOTAL_SALES_{col}" for col in df_pivoted.columns]
# 
# print(df_pivoted.head(20))
