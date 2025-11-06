import numpy as np
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ProphetTrainer:
    def __init__(self, model_params: dict, features, target):
        self.model_params = model_params
        self.model = Prophet(**model_params)
        self.features = features
        self.target = target
    
    def fit(self, df_train):
        self.model = Prophet(**self.model_params)
        for feature in self.features:
            self.model.add_regressor(feature)
        self.model.fit(df_train)

    def predict(self, df_test):
        preds = self.model.predict(df_test)
        yhat = preds["yhat"]
        yhat_interval = (preds["yhat_lower"], preds["yhat_upper"])
        return yhat, yhat_interval
    
    def eval(self, df_test) -> dict:
        y_test = df_test[self.target]
        yhat, yhat_interval = self.predict(df_test)
        metrics = {}
        metrics["total_rmse"] = np.sqrt(mean_squared_error(y_test, yhat)) 
        metrics["total_mae"] = mean_absolute_error(y_test, yhat)
        coverage_mask = (y_test >= yhat_interval[0]) & (y_test <= yhat_interval[1])
        metrics["coverage"] = coverage_mask.mean()
        if len(self.target) > 1:
            for idx, category in enumerate(self.target):
                metrics[f"{category}_rmse"] = np.sqrt(mean_squared_error(y_test.iloc[:, idx], yhat[:, idx])) 
                metrics[f"{category}_mae"] = mean_absolute_error(y_test.iloc[:, idx], yhat[:, idx])
                metrics[f"{category}_coverage"] = coverage_mask[:, idx].mean()

        return metrics
    
    def save_model(self, dir):
        with open(f"{dir}/xgb_model.json", 'w') as f:
            f.write(model_to_json(self.model))

    def load_model(self, dir: str, file_name: str="prophet_model.json"):
        with open(f"{dir}/{file_name}", 'r') as f:
            self.model = model_from_json(f.read())



    