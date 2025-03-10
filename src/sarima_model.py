import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_forecast(data_series, p=1, d=1, q=1, seasonal=True):
    seasonal_used = False
    
    try:
        n_obs = len(data_series)
        
        # Penyesuaian komponen musiman
        if seasonal:
            if n_obs >= 36:
                seasonal_order = (1, 1, 1, 12)
                seasonal_used = True
            elif 24 <= n_obs < 36:
                seasonal_order = (0, 1, 0, 12)  # Hanya differencing
                seasonal_used = "diff_only"
            else:
                seasonal_order = (0, 0, 0, 0)
                seasonal_used = False
        else:
            seasonal_order = (0, 0, 0, 0)
            seasonal_used = False

        # Konfigurasi model
        model = SARIMAX(
            data_series,
            order=(p, d, q),
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        
        # Latih model
        results = model.fit(disp=False, maxiter=200, method='powell')
        
        # Prediksi 12 bulan
        forecast = results.get_forecast(steps=12)
        forecast_ci = forecast.conf_int()
        
        # Format hasil
        forecast_df = pd.DataFrame({
            "Prediksi": forecast.predicted_mean,
            "Lower CI": forecast_ci.iloc[:, 0],
            "Upper CI": forecast_ci.iloc[:, 1]
        })
        
        return forecast_df, results, seasonal_used
    
    except Exception as e:
        return None, None, str(e)