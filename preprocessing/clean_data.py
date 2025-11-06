import os
from datetime import datetime, time

import pandas as pd


class DataCleaningPipeline:
    """
    A class that manages a sequence of data cleaning steps.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the pipeline with a DataFrame.
        """
        self.df = df
        self.steps = []

    def add_step(self, func, *args, **kwargs):
        """
        Adds a cleaning step to the pipeline.
        """
        self.steps.append((func, args, kwargs))

    def run(self) -> pd.DataFrame:
        """
        Runs all steps in the pipeline in order and returns the cleaned DataFrame.
        """
        for func, args, kwargs in self.steps:
            self.df = func(self.df, *args, **kwargs)
        return self.df


def fill_missing_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing time intervals for the past two years with valid time slots.
    """

    # list of all days and valid intervals for past two years
    all_dates = pd.date_range(start="2023-01-01", end="2024-12-31").date
    valid_intervals = ["11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45", "14:00"]

    full_time_range = [datetime.combine(date, datetime.strptime(time, "%H:%M").time()) 
                       for date in all_dates for time in valid_intervals]
  
    full_df = pd.DataFrame({'time': full_time_range})
    
    df = full_df.merge(df, on='datetime', how='left').fillna(0)
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(int)
    
    return df

def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Removes any rows outside of opening hours.
    """
    df["time"] = df["datetime"].dt.time
    df = df[(df["time"] >= time(11, 30)) & (df["time"] <= time(14, 0))]
    df.drop(columns=["time"], inplace=True)
    return df


def drop_zero_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that only contain zero values.
    """
    return df.loc[:, (df != 0).any(axis=0)]


def add_total_sales_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'total_sales' column as the sum of all sales columns.
    """
    sales_columns = df.columns.difference(['datetime'])  # All columns except 'time'
    
    # Add the total sales as the sum of all sales columns
    df["TOTAL_SALES"] = df[sales_columns].sum(axis=1)
    
    return df


def process_csv_file(path: str, csv_file: str) -> None:
    """
    Reads a CSV file, applies data cleaning steps, and saves the cleaned data.
    """
    df = pd.read_csv(f"{path}/{csv_file}", delimiter=',') 
    pipeline = DataCleaningPipeline(df)
    # change name of datetime column in place
    df.rename(columns={"time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # pipeline.add_step(drop_zero_columns)
    pipeline.add_step(add_total_sales_column)
    # pipeline.add_step(fill_missing_times)
    pipeline.add_step(clean_rows)
    
    cleaned_df = pipeline.run()

    cleaned_filename = f"cleaned_{csv_file}"  # Prepend 'cleaned_' to the file name

    cleaned_df.to_csv(f"{path}/{cleaned_filename}", index=False)  # Save cleaned_filename, index=False)
    print(f"Processed and saved cleaned data as {cleaned_filename}.")


# Example usage
# process_csv_file(path = "./data", csv_file = "store2.csv")