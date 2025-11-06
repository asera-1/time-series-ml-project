import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Define file paths
store1_path = r"D:\coxi\block_course\final-project-demand-prediction-super-awesome-team\data\processed_store1.csv"
store2_path = r"D:\coxi\block_course\final-project-demand-prediction-super-awesome-team\data\processed_store2.csv"
# Load processed data
df_store1 = pd.read_csv(store1_path)
df_store2 = pd.read_csv(store2_path)

# Rename the date column to match Prophet requirements
df_store1.rename(columns={'date': 'ds', 'total_sales': 'y'}, inplace=True)
df_store2.rename(columns={'date': 'ds', 'total_sales': 'y'}, inplace=True)

# Convert 'ds' column to datetime format
df_store1['ds'] = pd.to_datetime(df_store1['ds'], errors='coerce')
df_store2['ds'] = pd.to_datetime(df_store2['ds'], errors='coerce')

# Display missing values (if any)
print("Missing Dates in Store 1:", df_store1['ds'].isna().sum())
print("Missing Dates in Store 2:", df_store2['ds'].isna().sum())

# K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Define hyperparameter search space
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.05,0.025, 0.1, 0.5],
    'seasonality_prior_scale': [1, 2, 5, 10, 20,25 , 30],
    'holidays_prior_scale': [1, 2, 5, 10, 20]
}

all_params = list(ParameterGrid(param_grid))
best_mae = float("inf")
best_params = None

# Grid Search for best Prophet parameters
for params in all_params:
    mae_scores = []
    
    for train_index, test_index in kf.split(df_store1):
        train_store1, test_store1 = df_store1.iloc[train_index], df_store1.iloc[test_index]
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale']
        )
        model.fit(train_store1)
        
        future = pd.DataFrame({'ds': test_store1['ds']})
        forecast = model.predict(future)
        test_store1 = test_store1.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        
        mae = mean_absolute_error(test_store1['y'], test_store1['yhat'])
        mae_scores.append(mae)
    
    avg_mae = np.mean(mae_scores)
    if avg_mae < best_mae:
        best_mae = avg_mae
        best_params = params

# Train final Prophet model with best parameters
prophet_final = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    holidays_prior_scale=best_params['holidays_prior_scale']
)
prophet_final.fit(df_store1)

# Predict for the next 60 days
future_final = prophet_final.make_future_dataframe(periods=60)
forecast_final = prophet_final.predict(future_final)

# Plot actual vs. predicted sales
plt.figure(figsize=(12, 6))
plt.plot(df_store1['ds'], df_store1['y'], label="Actual Sales", alpha=0.5)
plt.plot(forecast_final['ds'], forecast_final['yhat'], label="Predicted Sales", color='red', alpha=0.7)
plt.fill_between(forecast_final['ds'], forecast_final['yhat_lower'], forecast_final['yhat_upper'], color='gray', alpha=0.2)
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Fine-Tuned Prophet Model: Forecast vs Actual")
plt.legend()
plt.show()

# Evaluate final model on last 60 days
df_test = df_store1[-60:]
forecast_test_final = forecast_final[-60:]

y_actual_final = df_test['y'].values
y_predicted_final = forecast_test_final['yhat'].values

# Compute MAE & RMSE
final_mae = mean_absolute_error(y_actual_final, y_predicted_final)
final_rmse = mean_squared_error(y_actual_final, y_predicted_final) ** 0.5

# Save results
final_results = pd.DataFrame({
    "Metric": ["MAE", "RMSE"],
    "Final Prophet Model": [final_mae, final_rmse]
})

# Print best parameters and final results
print("✅ Best Hyperparameters Found:")
print(best_params)
print("\n✅ Final Prophet Model Performance:")
print(final_results)

# Save evaluation results
results_dir = "D:/coxi/block_course/final-project-demand-prediction-super-awesome-team/results"
os.makedirs(results_dir, exist_ok=True)
final_results.to_csv(f"{results_dir}/prophet_final_results.csv", index=False)