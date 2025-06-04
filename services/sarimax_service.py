import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from flask import current_app
from utils import safe_filename

class SARIMAXForecaster:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.fit = None
        self.order = None
        self.seasonal_order = None
        self.family = None

    def load_and_preprocess(self, df, date_col, family_col, target_col, family_val):
        # Assume df is a Pandas DataFrame
        df[date_col] = pd.to_datetime(df[date_col])
        if family_val:
            df = df[df[family_col] == family_val]
        df = df.sort_values(date_col)
        self.family = family_val

                # --- Filter out missing or zero labels ---
        df = df[df[target_col].notnull()]
        df = df[df[target_col] != 0]

        # Aggregate daily
        ts_data = (
            df
            .groupby(date_col)[[target_col]]
            .sum()
            .reset_index()
        )
        ts_data.set_index(date_col, inplace=True)
        ts_data.index = pd.DatetimeIndex(ts_data.index).to_period('D')

        # Fill missing dates
        full_range = pd.period_range(ts_data.index.min(), ts_data.index.max(), freq='D')
        ts_data = ts_data.reindex(full_range, fill_value=0)

        # Target variable (log transform)
        ts_data['y'] = np.log1p(ts_data[target_col] + 1)

        # Features (day of week, month, etc.)
        ts_data['dow'] = ts_data.index.to_timestamp().dayofweek
        ts_data['month'] = ts_data.index.to_timestamp().month

        self.ts_data = ts_data

    def find_optimal_parameters(self):
        stepwise = auto_arima(self.ts_data['y'], seasonal=True, m=7, trace=False, error_action='ignore', suppress_warnings=True)
        self.order = stepwise.order
        self.seasonal_order = stepwise.seasonal_order

    def train_model(self):
        model = SARIMAX(
            self.ts_data['y'],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fit = model.fit(disp=False)
        # Save model
        family_name = self.family if self.family else "all"
        model_path = os.path.join(self.model_dir, f'sarimax_{family_name}.pkl')
        self.fit.save(model_path)
        return model_path

    def forecast(self, steps=30):
        forecast_result = self.fit.get_forecast(steps=steps)
        pred_log = forecast_result.predicted_mean
        pred_ci = forecast_result.conf_int()
        pred = np.expm1(pred_log) - 1
        pred_ci_orig = np.expm1(pred_ci) - 1
        return {
            "prediction": pred.tolist(),
            "lower_ci": pred_ci_orig.iloc[:, 0].tolist(),
            "upper_ci": pred_ci_orig.iloc[:, 1].tolist()
        }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['csv']

class SarimaxService:
    def __init__(self):
        self.model_dir = 'models'

    def train(self, file_storage, form):
        if not file_storage or not allowed_file(file_storage.filename):
            return {'error': 'No valid CSV file uploaded.'}

        date_col = form.get('date_col', 'date')
        family_col = safe_filename(form.get('family_col', 'family'))
        target_col = form.get('target_col', 'sales')
        family_val = safe_filename(form.get('family_val', None))

        df = pd.read_csv(file_storage)

        results = {}

        if not family_val:  # Train all families (batch)
            all_families = df[family_col].unique()
            for fam in all_families:
                try:
                    forecaster = SARIMAXForecaster(model_dir=self.model_dir)
                    forecaster.load_and_preprocess(df, date_col, family_col, target_col, fam)
                    forecaster.find_optimal_parameters()
                    model_path = forecaster.train_model()
                    results[fam] = f'Model saved at {model_path}'
                except Exception as e:
                    results[fam] = f'Error: {str(e)}'
            return {'message': 'Batch training complete', 'results': results}

        else:  # Train single family
            forecaster = SARIMAXForecaster(model_dir=self.model_dir)
            forecaster.load_and_preprocess(df, date_col, family_col, target_col, family_val)
            forecaster.find_optimal_parameters()
            model_path = forecaster.train_model()
            return {'message': f'SARIMAX model trained and saved at {model_path}.'}

    def predict(self, family_val, steps=30):
        # Load model and forecast
        model_path = os.path.join(self.model_dir, f'sarimax_{family_val}.pkl')
        if not os.path.exists(model_path):
            return {'error': f'No model found for family/item/category: {family_val}'}
        fit = SARIMAXResults.load(model_path)
        pred = fit.get_forecast(steps=steps).predicted_mean
        pred = np.expm1(pred) - 1
        return {'prediction': pred.tolist()}