from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FEATURE_DATA_DIR


def assign_holiday(dates: pd.Series) -> pd.Series:
    """
    Assigns 1 if holiday and 0 if not based on holiday dates (between 2023 and 2025)
    """
    dates = pd.to_datetime(dates).dt.date
    holidays_path = Path(FEATURE_DATA_DIR) / "niedersachsen_holidays.csv"
    holidays = pd.read_csv(holidays_path)
    holidays["date"] = pd.to_datetime(holidays["Tag"]).dt.date
    holiday_dates = set(holidays["date"].unique())
    
    return dates.isin(holiday_dates).astype(int)
 
def assign_open(dates: pd.Series) -> pd.Series:
    """
    Assigns 1 if open and 0 if not based on time
    """
    times = pd.to_datetime(dates).dt.time
    dates = pd.to_datetime(dates).dt.date
    opening_hours = pd.date_range("11:30", "14:00", freq="15min").time
    open = times.isin(opening_hours).astype(int)
    open = open * (1 - assign_holiday(dates))
    return open   

def assign_semester_period(dates: pd.Series, institution: str) -> pd.Series:
    """
    Assign semester time period (1 = Winter, 2 = Summer)
    
    Args:
        date: Date to check (pd.Series of date-like objects)
        institution: Name of educational institution
        
    Returns:
        int: 1 for Winter semester, 2 for Summer semester
    """
    dates = pd.to_datetime(dates)
    years = dates.dt.year
    semester_path = Path(FEATURE_DATA_DIR) / "semester_dates.csv"
    semester_df = pd.read_csv(semester_path)
    
    # Filter for institution
    institution_data = semester_df[semester_df["institution"] == institution]
    if institution_data.empty:
        raise ValueError(f"Institution '{institution}' not found in semester data")
    
    # Parse SS_begin and SS_end into datetime
    institution_data.loc[:, "SS_begin"] = pd.to_datetime(institution_data["SS_begin"])
    institution_data.loc[:, "SS_end"] = pd.to_datetime(institution_data["SS_end"])
    
    # Merge the dates with semester periods on 'year'
    merge_df = pd.DataFrame({"date": dates, "year": years})
    merged = merge_df.merge(institution_data, how='left', left_on='year', right_on='year')

    if merged.isnull().any().any():
        missing_years = merged[merged["SS_begin"].isna()]["year"].unique()
        raise ValueError(f"Year(s) {missing_years} not found for institution '{institution}'")

    # Check if date is between SS_begin and SS_end
    ss_mask = (merged["date"] >= merged["SS_begin"]) & (merged["date"] <= merged["SS_end"])
    
    # 2 if in summer semester, else 1 (winter semester)
    semester_period = np.where(ss_mask, 2, 1)
    
    return pd.Series(semester_period, index=dates.index)
        
def assign_semester_phase(dates: pd.Series, institution: str) -> pd.Series:
    """
    Function to assign semester phase (1 = Winter lectures, 2 = Summer lectures, 0 = Break)
    Args:
        dates: pd.Series of date-like objects
        institution: institution name
    """
    # Ensure dates are datetime
    dates = pd.to_datetime(dates)
    years = dates.dt.year

    # Load semester data
    semester_path = Path(FEATURE_DATA_DIR) / "semester_dates.csv"
    semester_df = pd.read_csv(semester_path)

    # Filter for institution
    institution_data = semester_df[semester_df["institution"] == institution].copy()
    if institution_data.empty:
        raise ValueError(f"Institution '{institution}' not found in semester data")

    # Convert date columns to datetime
    for col in ["SL_begin", "SL_end", "WL_begin", "WL_end", "Wb_begin", "Wb_end"]:
        institution_data.loc[:, col] = pd.to_datetime(institution_data[col])

    # Merge the dates with semester schedule on 'year'
    merge_df = pd.DataFrame({"date": dates, "year": years})
    merged = merge_df.merge(institution_data, how="left", on="year")

    if merged.isnull().any().any():
        missing_years = merged[merged["SL_begin"].isna()]["year"].unique()
        raise ValueError(f"Year(s) {missing_years} not found for institution '{institution}'")

    # Vectorized conditions
    sl_mask = (merged["date"] >= merged["SL_begin"]) & (merged["date"] <= merged["SL_end"])
    wl_pre_mask = (merged["date"] >= merged["WL_begin"]) & (merged["date"] < merged["Wb_begin"])
    wl_post_mask = (merged["date"] > merged["Wb_end"]) & (merged["date"] <= merged["WL_end"])

    # Assign semester phase (default is 0 for break)
    semester_phase = np.select([sl_mask, wl_pre_mask, wl_post_mask], [1, 2, 2], default=0)

    return pd.Series(semester_phase, index=dates.index)

def uni_semester_phase(dates: pd.Series)-> pd.Series:
    """
    Assigns semester phase (1 = Winter lectures, 2 = Summer lectures, 0 = break)
    """
    return assign_semester_phase(dates, "uni")

def hs_semester_phase(dates: pd.Series)-> pd.Series:
    """
    Assigns semester phase (1 = Winter lectures, 2 = Summer lectures, 0 = break)
    """
    return assign_semester_phase(dates, "hs")

def uni_semester_period(dates: pd.Series)-> pd.Series:
    """
    Assigns semester time period (1 = Winter, 2 = Summer)
    """
    return assign_semester_period(dates, "uni")

def hs_semester_period(dates: pd.Series)-> pd.Series:
    """
    Assigns semester time period (1 = Winter, 2 = Summer)
    """
    return assign_semester_period(dates, "hs")