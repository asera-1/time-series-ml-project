import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


class XGBTrainer:
    def __init__(self, model_params: dict, features, target):
        self.model_params = model_params
        self.model = xgb.XGBRegressor(**model_params)
        self.features = features
        self.target = target
    
    def fit(self, df_train):
        X = df_train[self.features]
        y = df_train[self.target]
        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(X, y)

    def predict(self, df_test):
        X = df_test[self.features]
        yhat = self.model.predict(X)
        return yhat

    def eval(self, df_test)-> dict:
        X_test = df_test[self.features]
        y_test = df_test[self.target]
        preds = self.predict(X_test)
        metrics = {}
        if len(self.target) > 1:
            metrics["total_rmse"] = np.sqrt(mean_squared_error(y_test, preds)) 
            metrics["total_mae"] = mean_absolute_error(y_test, preds)
            for idx, category in enumerate(self.target):
                metrics[f"{category}_rmse"] = np.sqrt(mean_squared_error(y_test.iloc[:, idx], preds[:, idx]))
                metrics[f"{category}_mae"] = mean_absolute_error(y_test.iloc[:, idx], preds[:, idx])
        else:
            metrics[f"{self.target[0]}_rmse"] = np.sqrt(mean_squared_error(y_test, preds))
            metrics[f"{self.target[0]}_mae"] = mean_absolute_error(y_test, preds) 
        for feat, imp in zip(self.features, self.model.feature_importances_):
            metrics[f"{feat}_importance"] = imp

        return metrics

    def save_model(self, dir: str):
        self.model.save_model(f"{dir}/xgb_model.json")
    
    def load_model(self, dir: str, file_name: str="xgb_model.json"):
        self.model = xgb.XGBRegressor()
        self.model.load_model(f"{dir}/{file_name}")
