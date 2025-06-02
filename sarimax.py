# ========== IMPORT LIBRARY ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# ========== LOAD DAN EKSPLORASI DATA ==========
print("Loading and exploring data...")
df = pd.read_csv('Fashion_Retail_Sales.csv')
print("\nSample data:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nData description:")
print(df.describe(include="all"))

# ========== DATA CLEANING ==========
print("\nCleaning data...")
df = df.dropna(subset=['Purchase Amount (USD)'])
df['Review Rating'] = df['Review Rating'].fillna(df['Review Rating'].mean())
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'])
df = df.sort_values('Date Purchase')

# ========== PREPROCESSING DATA ==========
print("\nPreprocessing data...")
df_daily = df.set_index('Date Purchase').resample('D').agg({
    'Purchase Amount (USD)': 'sum',
    'Item Purchased': 'count',
    'Review Rating': 'mean'
}).fillna(0)

# Smoothing
df_daily['Purchase Amount (USD)'] = df_daily['Purchase Amount (USD)'].rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')

# Time-based features
df_daily['day_of_week'] = df_daily.index.dayofweek
df_daily['month'] = df_daily.index.month
df_daily['quarter'] = df_daily.index.quarter
df_daily['is_weekend'] = df_daily['day_of_week'].isin([5,6]).astype(int)
df_daily['MA7'] = df_daily['Purchase Amount (USD)'].rolling(window=7).mean()
df_daily['MA30'] = df_daily['Purchase Amount (USD)'].rolling(window=30).mean()
df_daily = df_daily.fillna(method='bfill').fillna(method='ffill')

exog_columns = ['Item Purchased', 'Review Rating', 'day_of_week', 'month', 'quarter', 'is_weekend', 'MA7', 'MA30']

# Scaling exogenous features
exog_scaler = MinMaxScaler()
df_daily[exog_columns] = exog_scaler.fit_transform(df_daily[exog_columns])

# Scaling target
scaler = MinMaxScaler()
scaled_purchase = scaler.fit_transform(df_daily[['Purchase Amount (USD)']])
df_daily['Purchase Amount (USD)'] = scaled_purchase

# ========== TRAIN-VAL-TEST SPLIT (70/20/10) ==========
print("\nSplitting data...")
n = len(df_daily)
train_size = int(n * 0.7)
val_size = int(n * 0.2)
test_size = n - train_size - val_size

train_data = df_daily.iloc[:train_size]
val_data = df_daily.iloc[train_size:train_size+val_size]
test_data = df_daily.iloc[train_size+val_size:]

train_exog = train_data[exog_columns]
val_exog = val_data[exog_columns]
test_exog = test_data[exog_columns]

# ========== SARIMAX PARAMETER TUNING & TRAINING ==========
print("\nTraining SARIMAX Model with Exogenous Variables...")
param_combinations = [
    ((1,1,1), (1,1,1,7)),
    ((2,1,2), (1,1,1,7)),
    ((1,1,2), (0,1,1,7)),
    ((2,0,2), (1,1,1,7)),
    ((1,1,1), (2,1,1,7)),
]
best_aic = float('inf')
best_sarimax_model = None
best_params = None

for order, seasonal_order in param_combinations:
    try:
        print(f"Testing SARIMAX parameters: {order}, {seasonal_order}")
        model = SARIMAX(
            train_data['Purchase Amount (USD)'],
            exog=train_exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend='c'
        )
        result = model.fit(disp=False)
        if result.aic < best_aic:
            best_aic = result.aic
            best_sarimax_model = result
            best_params = (order, seasonal_order)
            print(f"  New best AIC: {best_aic:.2f}")
    except Exception as e:
        print(f"  Error with params {order}, {seasonal_order}: {str(e)[:100]}...")
        continue

if best_sarimax_model is None:
    print("Fallback SARIMAX used.")
    model = SARIMAX(
        train_data['Purchase Amount (USD)'],
        exog=train_exog,
        order=(1,1,1),
        seasonal_order=(1,1,1,7),
        trend='c'
    )
    best_sarimax_model = model.fit(disp=False)
    best_params = ((1,1,1), (1,1,1,7))

print(f"\nBest SARIMAX params: {best_params}, AIC: {best_sarimax_model.aic:.2f}")
print("SARIMAX exogenous variables:", exog_columns)

# ========== SARIMAX PREDICTION ==========
print("\nGenerating SARIMAX predictions...")
forecast = best_sarimax_model.get_forecast(steps=len(test_data), exog=test_exog)
sarimax_pred = forecast.predicted_mean
sarimax_pred_lower = forecast.conf_int().iloc[:, 0]
sarimax_pred_upper = forecast.conf_int().iloc[:, 1]

# Inverse scaling
sarimax_pred = scaler.inverse_transform(sarimax_pred.values.reshape(-1,1)).flatten()
sarimax_pred_lower = scaler.inverse_transform(sarimax_pred_lower.values.reshape(-1,1)).flatten()
sarimax_pred_upper = scaler.inverse_transform(sarimax_pred_upper.values.reshape(-1,1)).flatten()
actual_test = scaler.inverse_transform(test_data['Purchase Amount (USD)'].values.reshape(-1,1)).flatten()

# ========== EVALUASI SARIMAX ==========
mae_sarimax = mean_absolute_error(actual_test, sarimax_pred)
mape_sarimax = mean_absolute_percentage_error(actual_test, sarimax_pred) * 100
r2_sarimax = r2_score(actual_test, sarimax_pred)
rmse_sarimax = np.sqrt(mean_squared_error(actual_test, sarimax_pred))

print("\n=== SARIMAX Evaluation ===")
print(f"RMSE  : {rmse_sarimax:.2f}")
print(f"MAE   : {mae_sarimax:.2f}")
print(f"MAPE  : {mape_sarimax:.2f}%")
print(f"R2    : {r2_sarimax:.4f}")

# ========== VISUALISASI HASIL PREDIKSI SARIMAX ==========
print("\nVisualizing SARIMAX results...")
plt.figure(figsize=(15,7))
plt.plot(test_data.index, actual_test, label='Actual', linewidth=2, color='black')
plt.plot(test_data.index, sarimax_pred, label='SARIMAX Prediction', linewidth=2, color='blue', alpha=0.8)
plt.fill_between(test_data.index, sarimax_pred_lower, sarimax_pred_upper, 
                 color='blue', alpha=0.1, label='SARIMAX 95% CI')
plt.title('SARIMAX Model Prediction Results', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Purchase Amount (USD)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.text(0.02, 0.95, f'RMSE: {rmse_sarimax:.2f}\nR²: {r2_sarimax:.4f}', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()

# ========== VISUALISASI METRIK PERFORMA SARIMAX ==========
print("\nVisualizing SARIMAX performance metrics...")
plt.figure(figsize=(10, 6))
metrics = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
sarimax_values = [rmse_sarimax, mae_sarimax, mape_sarimax, r2_sarimax*100]
x = np.arange(len(metrics))
width = 0.5
bars = plt.bar(x, sarimax_values, width, label='SARIMAX', color='blue', alpha=0.7)

for i, bar in enumerate(bars):
    if i == 3:  # R² Score
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{sarimax_values[i]/100:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{sarimax_values[i]:.1f}', ha='center', va='bottom', fontweight='bold')

plt.title('SARIMAX Model Performance Metrics', fontsize=16, fontweight='bold')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.figtext(0.5, 0.02,
           'Note: For RMSE, MAE, MAPE → lower is better | For R² → higher is better',
           ha='center', fontsize=10, style='italic')
plt.tight_layout()
plt.show()

# ========== RINGKASAN PERFORMA SARIMAX ==========
print(f"\n{'='*50}")
print("RINGKASAN PERFORMA SARIMAX:")
print(f"{'='*50}")
print(f"Best parameters: {best_params}")
print(f"RMSE: {rmse_sarimax:.2f}")
print(f"MAE: {mae_sarimax:.2f}")
print(f"MAPE: {mape_sarimax:.2f}%")
print(f"R²: {r2_sarimax:.4f}")
print(f"{'='*50}")

# Optional: Save the model and results
import pickle
print("\nSaving SARIMAX model and results...")
with open('sarimax_model_results.pkl', 'wb') as f:
    pickle.dump({
        'model': best_sarimax_model,
        'params': best_params,
        'predictions': sarimax_pred,
        'actual': actual_test,
        'metrics': {
            'rmse': rmse_sarimax,
            'mae': mae_sarimax,
            'mape': mape_sarimax,
            'r2': r2_sarimax
        }
    }, f)
print("SARIMAX model and results saved to 'sarimax_model_results.pkl'")