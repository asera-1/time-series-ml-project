import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.prophet
import json
import os


mlflow.set_tracking_uri("https://mlflow-aip-2.visiolab.io")


os.environ["MLFLOW_TRACKING_USERNAME"] = "aip-2"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "5cnoAqn8DntcAZPjVPKp5hsb6wYGluu3hNXCIRJKpJPdr8q4OKl3pxFyH1PMqoB0"

store1_path = r"D:\coxi\block_course\final-project-demand-prediction-super-awesome-team\data\processed_store1.csv"
store2_path = r"D:\coxi\block_course\final-project-demand-prediction-super-awesome-team\data\processed_store2.csv"

store_id = int(input("Enter Store ID (1 or 2): "))
data_path = store1_path if store_id == 1 else store2_path

df = pd.read_csv(data_path)
df['ds'] = pd.to_datetime(df['date'])
df = df.groupby('ds').agg({'total_sales': 'sum'}).reset_index()
df.rename(columns={'total_sales':'y'}, inplace=True)

def assign_semester_phase(date):
    if pd.Timestamp("2024-10-01") <= date <= pd.Timestamp("2025-03-31"):
        return 1 # Winter Semester
    elif pd.Timestamp("2025-04-01") <= date <= pd.Timestamp("2025-09-30"):
        return 2 # Summer Semester
    return 0

df['semester_phase'] = df['ds'].apply(assign_semester_phase)

holidays = pd.DataFrame({
    'holiday': 'semester_break',
    'ds': pd.date_range(start='2024-07-15', end='2024-09-15', freq='D'),
    'lower_window': 0,
    'upper_window': 1
})

christmas_break = pd.DataFrame({
    'holiday': 'christmas_break',
    'ds': pd.date_range('2024-12-22', '2025-01-02'),
    'lower_window': 0,
    'upper_window': 1
})

holidays = pd.concat([holidays, christmas_break])

# Set MLflow experiment 
mlflow.set_experiment(experiment_name="Prophet Demand Prediction")

with mlflow.start_run(run_name=f"store_{store_id}_prophet_baseline"):
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, holidays=holidays)
    model.add_country_holidays(country_name='DE')
    model.add_regressor('semester_phase')

    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses = [], []

    print("🚀 Starting Time Series Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        train_fold, val_fold = df.iloc[train_idx], df.iloc[val_idx]

        model_fold = Prophet(yearly_seasonality=True, weekly_seasonality=True, holidays=holidays)
        model_fold.add_country_holidays(country_name='DE')
        model_fold.add_regressor('semester_phase')
        
        model_fold.fit(train_fold)

        future = val_fold[['ds', 'semester_phase']]
        forecast_fold = model_fold.predict(future)

        mae_fold = mean_absolute_error(val_fold['y'], forecast_fold['yhat'])
        rmse_fold = np.sqrt(mean_squared_error(val_fold['y'], forecast_fold['yhat']))

        maes.append(mae_fold)
        rmses.append(rmse_fold)

        print(f"Fold {fold+1} - MAE: {mae_fold:.2f}, RMSE: {rmse_fold:.2f}")

    avg_mae, avg_rmse = np.mean(maes), np.mean(rmses)

    print("\n Cross-Validation Results:")
    print(f"Average MAE: {avg_mae:.2f}")
    print(f"Average RMSE: {avg_rmse:.2f}")

    model.fit(df)

    # Log the model and metrics in MLflow
    mlflow.prophet.log_model(model, "prophet-model")
    mlflow.log_param("store_id", store_id)
    mlflow.log_metric("avg_mae", avg_mae)
    mlflow.log_metric("avg_rmse", avg_rmse)

    os.makedirs("models", exist_ok=True)
    with open(f'models/prophet_final_store{store_id}.json', 'w') as fout:
        json.dump(model_to_json(model), fout)

print("\n Model training complete and logged to MLflow at https://mlflow-aip-2.visiolab.io!")
