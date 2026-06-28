import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

try:
    from nixtla import NixtlaClient
except ImportError:
    NixtlaClient = None

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

load_dotenv()
TIMEGPT_TOKEN = os.getenv("TIMEGPT_TOKEN")

def get_forecast_dataframe(df_monthly: pd.DataFrame, horizon: int = 3) -> tuple[pd.DataFrame, str, str]:
    """
    Renvoie un DataFrame combinant l'historique et les prévisions, ainsi que le nom du modèle et le statut.
    """
    if len(df_monthly) < 3:
        df_hist = df_monthly.copy()
        df_hist['Type'] = 'Historique'
        return df_hist, 'None', 'insufficient_data'
        
    df_ts = df_monthly[['Date', 'Quantite']].rename(columns={'Date': 'ds', 'Quantite': 'y'}).copy()
    
    # 1. Essayer TimeGPT
    if TIMEGPT_TOKEN and NixtlaClient:
        try:
            nixtla_client = NixtlaClient(api_key=TIMEGPT_TOKEN)
            
            # Validation du token
            if nixtla_client.validate_api_key():
                df_ts['ds'] = pd.to_datetime(df_ts['ds'])
                
                # Forecasting
                fcst_df = nixtla_client.forecast(
                    df=df_ts,
                    h=horizon,
                    time_col='ds',
                    target_col='y',
                    freq='MS'
                )
                
                fcst_df = fcst_df.rename(columns={'ds': 'Date', 'TimeGPT': 'Quantite'})
                fcst_df['Date'] = pd.to_datetime(fcst_df['Date']) + pd.offsets.MonthEnd(0)
                fcst_df['Quantite'] = fcst_df['Quantite'].apply(lambda x: max(0, int(x)))
                fcst_df['Type'] = 'Prévision'
                
                df_hist = df_monthly.copy()
                df_hist['Type'] = 'Historique'
                
                df_combined = pd.concat([df_hist, fcst_df], ignore_index=True)
                return df_combined, 'TimeGPT', 'success'
        except Exception as e:
            print(f"Erreur TimeGPT: {e}. Repli sur Prophet.")
            
    # 2. Repli sur Prophet
    if Prophet is not None:
        try:
            yearly_seasonality = 'auto' if len(df_ts) >= 12 else False
            
            m = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            m.fit(df_ts)
            future = m.make_future_dataframe(periods=horizon, freq='ME')
            forecast = m.predict(future)
            
            fcst_df = forecast.tail(horizon)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Quantite'})
            fcst_df['Date'] = pd.to_datetime(fcst_df['Date']) + pd.offsets.MonthEnd(0)
            fcst_df['Quantite'] = fcst_df['Quantite'].apply(lambda x: max(0, int(x)))
            fcst_df['Type'] = 'Prévision'
            
            df_hist = df_monthly.copy()
            df_hist['Type'] = 'Historique'
            
            df_combined = pd.concat([df_hist, fcst_df], ignore_index=True)
            return df_combined, 'Prophet', 'success'
        except Exception as e:
            print(f"Erreur Prophet: {e}")
            df_hist = df_monthly.copy()
            df_hist['Type'] = 'Historique'
            return df_hist, 'Prophet', f'error: {e}'
            
    # 3. Dernier repli : Moyenne Mobile
    last_y = list(df_ts['y'].values)
    last_date = pd.to_datetime(df_ts['ds'].iloc[-1])
    
    # Generate future dates
    future_dates = []
    for i in range(1, horizon + 1):
        # Adding months safely
        next_month = last_date + pd.DateOffset(months=i)
        future_dates.append(next_month)
        
    preds = []
    for _ in range(horizon):
        if len(last_y) >= 3:
            val = int(np.average(last_y[-3:], weights=np.array([1, 2, 3])))
        else:
            val = int(np.mean(last_y))
        preds.append(val)
        last_y.append(val)
        
    fcst_df = pd.DataFrame({'Date': future_dates, 'Quantite': preds})
    fcst_df['Date'] = pd.to_datetime(fcst_df['Date']) + pd.offsets.MonthEnd(0)
    fcst_df['Type'] = 'Prévision'
    
    df_hist = df_monthly.copy()
    df_hist['Type'] = 'Historique'
    
    df_combined = pd.concat([df_hist, fcst_df], ignore_index=True)
    return df_combined, 'Moyenne Mobile', 'success'

def run_forecast(df_monthly: pd.DataFrame, horizon: int = 1) -> dict:
    """
    Exécute la prévision de séries temporelles.
    Utilise TimeGPT si un token est disponible, sinon se rabat sur Prophet (amélioré).
    
    Retourne un dictionnaire avec :
    - 'prediction': La valeur de prévision pour le mois M+horizon (int)
    - 'model_used': Le nom du modèle utilisé ('TimeGPT' ou 'Prophet')
    - 'status': 'success' ou message d'erreur
    """
    df_combined, model_used, status = get_forecast_dataframe(df_monthly, horizon)
    
    if status == 'success' or status.startswith('error'):
        df_pred = df_combined[df_combined['Type'] == 'Prévision']
        if not df_pred.empty:
            # We want the prediction at M+horizon, which is the last row since we predict `horizon` periods
            pred_value = int(df_pred.iloc[-1]['Quantite'])
        else:
            pred_value = 0
    else:
        pred_value = 0
        
    return {'prediction': pred_value, 'model_used': model_used, 'status': status}
