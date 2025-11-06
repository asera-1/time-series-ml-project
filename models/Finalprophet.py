import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
import matplotlib.pyplot as plt

store_paths = {
    "1": r"D:\coxi\block_course\final-project-demand-prediction-super-awesome-team\data\processed_store1.csv",
    "2": r"D:\coxi\block_course\final-project-demand-prediction-super-awesome-team\data\processed_store2.csv"
}

def load_store_data(store_id):
    df = pd.read_csv(store_paths[store_id])
    df['ds'] = pd.to_datetime(df['date'])
    df = df.groupby('ds')['total_sales'].sum().reset_index()
    df.rename(columns={'total_sales': 'y'}, inplace=True)
    return df

def assign_semester_phase(date):
    if ((date.month == 12 and date.day >= 22) or (date.month == 1 and date.day <= 2)):
        return 0  # Christmas break
    if (pd.Timestamp("2024-07-31") <= date <= pd.Timestamp("2024-09-15")):
        return 0  # Summer semester break
    if ((date.month >= 10) or (date.month <= 3)):
        return 1  # Winter semester
    elif (4 <= date.month <= 7):
        return 2  # Summer semester
    else:
        return 0  # Semester break

# Add semester phase
def add_semester_phase(df):
    df['semester_phase'] = df['ds'].apply(assign_semester_phase)
    return df

def create_custom_holidays():
    custom_holidays = pd.DataFrame({
        'holiday': 'Semester_Break',
        'ds': pd.date_range('2024-07-31', '2024-09-15'),
        'lower_window': 0,
        'upper_window': 0
    })
    christmas_break = pd.DataFrame({
        'holiday': 'Christmas_Break',
        'ds': pd.date_range('2024-12-22', '2025-01-02'),
        'lower_window': 0,
        'upper_window': 0
    })
    return pd.concat([custom_holidays, christmas_break])

# Prophet model training
def train_prophet(df):
    holidays = make_holidays_df(year_list=[2023, 2024, 2025], country='DE')
    holidays = pd.concat([holidays, create_custom_holidays()])
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=0.02,
        seasonality_prior_scale=20,
        holidays_prior_scale=1
    )
    model.add_regressor('semester_phase')
    model.fit(df)
    return model

def k_fold_validation(df, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_list, rmse_list = [], []

    fold = 1
    for train_idx, test_idx in tscv.split(df):
        train_fold, test_fold = df.iloc[train_idx], df.iloc[test_idx]

        model = train_prophet(train_fold)

        forecast_fold = model.predict(test_fold)
        mae = mean_absolute_error(test_fold['y'], forecast_fold['yhat'])
        rmse = np.sqrt(mean_squared_error(test_fold['y'], forecast_fold['yhat']))

        mae_list.append(mae)
        rmse_list.append(rmse)

        print(f"Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}")
        fold += 1

    print(f"\n Average MAE: {np.mean(mae_list):.2f}")
    print(f" Average RMSE: {np.mean(rmse_list):.2f}")


# Predict total sales for a given date and store
def predict_sales(store_id, date_input):
    df = load_store_data(store_id)
    df = add_semester_phase(df)
    model = train_prophet(df)

    future = pd.DataFrame({'ds': [pd.to_datetime(date_input)]})
    future = add_semester_phase(future)

    forecast = model.predict(future)
    predicted_sales = forecast.loc[0, 'yhat']

    print(f"\n📅 Predicted Total Sales on {date_input} for Store {store_id}: {predicted_sales:.2f}")

# Main execution
if __name__ == "__main__":
    store_id = input("Enter Store ID (1 or 2): ")
    df_store = load_store_data(store_id)
    df_store = add_semester_phase(df_store)

    print("\n Performing K-Fold Cross-Validation (last 90 days)...")
    validation_period = 90
    df_train = df_store[:-validation_period]
    df_test = df_store[-validation_period:]

    k_fold_validation(df_train, n_splits=8)

    print("\n Evaluating Final Model (last 90 days)...")
    model_final = train_prophet(df_train)
    forecast_final = model_final.predict(df_test)

    mae_final = mean_absolute_error(df_test['y'], forecast_final['yhat'])
    rmse_final = np.sqrt(mean_squared_error(df_test['y'], forecast_final['yhat']))
    print(f"\n Final Evaluation MAE: {mae_final:.2f}, RMSE: {rmse_final:.2f}")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(df_test['ds'], df_test['y'], label='Actual', color='blue')
    plt.plot(df_test['ds'], forecast_final['yhat'], label='Predicted', color='red', linestyle='--')
    plt.fill_between(df_test['ds'], forecast_final['yhat_lower'], forecast_final['yhat_upper'], color='gray', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.title('Prophet Forecast vs Actual (Last 90 Days)')
    plt.legend()
    plt.grid(True)
    plt.show()

    date_input = input("\nEnter date (YYYY-MM-DD) for sales prediction: ")
    predict_sales(store_id, date_input)
