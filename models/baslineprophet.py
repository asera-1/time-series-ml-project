import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("final_store2_data.csv")
df.rename(columns={'time': 'ds', 'total_sales': 'y'}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])

# Define semester phases without explicitly marking breaks
def assign_semester_phase(date):
    """ Assigns semester time periods (1 = Winter, 2 = Summer). """
    if ((date.month >= 10 and date.day >= 1) or (date.month <= 3 and date.day <= 31)):
        return 1  # Winter Semester
    elif (date.month >= 4 and date.day >= 1) or (date.month <= 7 and date.day <= 30):
        return 2  # Summer Semester
    return 0  # Unknown 

df['semester_phase'] = df['ds'].apply(assign_semester_phase)
df['semester_phase'] = df['semester_phase'].rolling(window=21, center=True).mean().fillna(method='bfill').fillna(method='ffill')  #  Increase smoothing window to 21 days

# Improved Outlier Handling using Robust Scaling (Winsorization)
def robust_outlier_cap(series, lower_percentile=1, upper_percentile=99):
    """ Winsorizes extreme values based on percentiles to avoid hard clipping. """
    lower_bound, upper_bound = np.percentile(series, [lower_percentile, upper_percentile])
    return np.clip(series, lower_bound, upper_bound)

df['y'] = robust_outlier_cap(df['y'])  # Enhanced outlier handling

df['y_smoothed'] = df['y'].rolling(window=7, center=True).median()
df['y'] = np.where(df['y_smoothed'].notna(), df['y_smoothed'], df['y'])
df.drop(columns=['y_smoothed'], inplace=True)

# Initialize Prophet Model with refined parameters
prophet_tuned = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.025,
    seasonality_prior_scale=15,
    holidays_prior_scale=7
)

# Add custom daily seasonality with controlled Fourier terms and emester_phase as a regressor
prophet_tuned.add_seasonality(name='daily', period=1, fourier_order=2)
prophet_tuned.add_regressor('semester_phase')

# Train and Save the model
prophet_tuned.fit(df[['ds', 'y', 'semester_phase']])
joblib.dump(prophet_tuned, "prophet_model.pkl")

# Perform Cross-Validation with adjusted parameters
data_length_days = (df['ds'].max() - df['ds'].min()).days
initial_period = '400 days' if data_length_days >= 400 else ('365 days' if data_length_days >= 365 else str(data_length_days - 60) + ' days')

df_cv = cross_validation(prophet_tuned, horizon='30 days', period='15 days', initial=initial_period)
df_performance = performance_metrics(df_cv)
print("Cross-Validation Performance Metrics:")
available_columns = ['horizon', 'mse', 'rmse', 'mae']


# Make predictions for the next 60 days
future_tuned = prophet_tuned.make_future_dataframe(periods=60)
future_tuned['semester_phase'] = future_tuned['ds'].apply(assign_semester_phase)
future_tuned['semester_phase'] = future_tuned['semester_phase'].rolling(window=21, center=True).mean().fillna(method='bfill').fillna(method='ffill')

forecast_tuned = prophet_tuned.predict(future_tuned)

# apply median filter to stabilize fluctuations in forecast
def apply_median_filter(series, window=5):
    return series.rolling(window=window, center=True).median().fillna(method='bfill').fillna(method='ffill')

forecast_tuned['yhat'] = apply_median_filter(forecast_tuned['yhat'], window=5)

# evaluate model on last 60 days
df_test = df[-60:]
forecast_test_tuned = forecast_tuned[-60:]

y_actual_tuned = df_test['y'].values
y_predicted_tuned = forecast_test_tuned['yhat'].values

# Compute error metrics
mae_tuned = mean_absolute_error(y_actual_tuned, y_predicted_tuned)
rmse_tuned = mean_squared_error(y_actual_tuned, y_predicted_tuned) ** 0.5

results_dict = {
    "Metric": ["MAE", "RMSE"],
    "Tuned Prophet Model": [mae_tuned, rmse_tuned]
}


pd.DataFrame(results_dict).to_csv("fine_tuned_prophet_results_with_semester_phase.csv", index=False)

prophet_tuned.plot(forecast_tuned)
plt.title("📊 Fine-Tuned Prophet Model (With Semester Timing, Outlier Handling & Smoother Seasonality)")
plt.show()

forecast_tuned.to_csv("prophet_forecast_results.csv", index=False)
