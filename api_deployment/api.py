import logging
from datetime import datetime, timedelta

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from demandprediction.models.reconciliation_hts.reconciliation import To_Reconcile
from demandprediction.preprocessing import calendar_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

historical_data = pd.read_csv("data/cleaned_adjusted_store1.csv")
historical_data["time"] = pd.to_datetime(historical_data["time"])
prophet_model = joblib.load("prophet_model.pkl")

class TimestampInput(BaseModel):
    """Model for input data containing a timestamp string."""
    timestamp: str  # Expecting an ISO formatted timestamp

class SalesPredictionService:
    """Service that processes timestamps, extracts features, and predicts sales."""

    def __init__(self, historical_data: pd.DataFrame, model):
        self.historical_data = historical_data
        self.model = model

    def validate_ios_timestamp(self, timestamp: str) -> datetime:
        """Validates and converts an ISO formatted timestamp string to a datetime object."""
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid iOS timestamp format.")

    def closest_previous_interval(self, user_time: datetime) -> tuple:
        """Finds the closest previous 15-minute interval before the given time."""
        valid_intervals = [
            "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", 
            "13:00", "13:15", "13:30", "13:45", "14:00"
        ]
        previous_intervals = [
            datetime.strptime(f"{user_time.date()} {t}:00", "%Y-%m-%d %H:%M:%S") for t in valid_intervals
        ]
        previous_intervals = [t for t in previous_intervals if t <= user_time]

        closest_time = max(previous_intervals) if previous_intervals else datetime.combine(
            user_time.date() - timedelta(days=1), datetime.strptime("14:00", "%H:%M").time()
        )
        end_time = closest_time + timedelta(minutes=15)
        return closest_time, end_time

    def extract_features(self, timestamp: datetime) -> dict:
        """Extracts time-based features and metadata for the given timestamp."""
        features = {
            "year": timestamp.year,
            "month": timestamp.month,
            "week": timestamp.isocalendar()[1],  # ISO week number
            "day": timestamp.day,
            "weekday": timestamp.weekday(),  # Monday = 0, Sunday = 6
            "hour": timestamp.hour
        }

        # Find if the timestamp falls on a holiday (assuming historical data contains a 'holiday' column)
        features["holiday"] = calendar_features.assign_holiday(pd.Series([timestamp]))[0]
        features["open"] = calendar_features.assign_open(pd.Series([timestamp]))[0]

        # Find the semester period (1 = Winter, 2 = Summer)
        features["semester_uni"] = calendar_features.assign_semester_period(pd.Series([timestamp]), "uni")[0]
        features["semester_hs"] = calendar_features.assign_semester_period(pd.Series([timestamp]), "hs")[0]
        features["semester_phase_uni"] = calendar_features.assign_semester_phase(pd.Series([timestamp]), "uni")[0]
        features["semester_phase_hs"] = calendar_features.assign_semester_phase(pd.Series([timestamp]), "hs")[0]

        return features

    def get_prophet_sales_prediction(self, timestamp: datetime) -> float:
        """Predicts sales based on historical data or trained model."""
        semester_phase = calendar_features.assign_semester_phase(pd.Series([timestamp]), "uni")[0]

        if timestamp < datetime(2024, 10, 1):
            historical_match = self.historical_data[self.historical_data["time"] == timestamp]
            if not historical_match.empty:
                return None  # No historical data available
        else:
            # Use Prophet model for future predictions
            future_df = pd.DataFrame({"ds": [timestamp]})
            future_df['semester_phase'] = semester_phase  # Include semester phase as a regressor
            prediction = self.model.predict(future_df)
            return prediction["yhat"].values[0]
        
    def get_xgboost_sales_prediction(self, timestamp: datetime) -> float:
        # Use XGBoost model for future predictions
        pass

    def temporal_reconciliation(self, timestamp: datetime):
        features = self.extract_features(timestamp)
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

sales_prediction_service = SalesPredictionService(historical_data, prophet_model)

@app.post("/process-timestamp/")
def process_timestamp(input_data: TimestampInput):
    """
    Processes an iOS timestamp, extracts features, finds the closest interval, and fetches sales prediction.
    """
    try:
        user_time = sales_prediction_service.validate_ios_timestamp(input_data.timestamp)
        start_time, end_time = sales_prediction_service.closest_previous_interval(user_time)

        logger.info(f"Validated timestamp: {user_time}")
        logger.info(f"Closest previous interval: {start_time} to {end_time}")

        # Extract features
        extracted_features = sales_prediction_service.extract_features(start_time)

        # Get sales prediction
        sales_prediction = sales_prediction_service.get_prophet_sales_prediction(start_time)
        if sales_prediction is None:
            sales_prediction = "No historical data available"

        return {
            "input_timestamp": input_data.timestamp,
            "validated_timestamp": user_time.isoformat(),
            "closest_interval_start": start_time.isoformat(),
            "closest_interval_end": end_time.isoformat(),
            "extracted_features": extracted_features,
            "sales_prediction": sales_prediction,
        }
    except HTTPException as e:
        logger.error(f"Error: {e.detail}")
        raise e


# To run:
# uvicorn api_draft2:app --reload
# Or use the HTTP POST request to:
# http://127.0.0.1:8000/process-timestamp/ with timestamp="2025-03-12T14:44:00"
