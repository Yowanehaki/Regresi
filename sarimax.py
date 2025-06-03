# ========== IMPORT LIBRARY ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

# ========== DATA CLEANING & PREPROCESSING ==========
print("\nCleaning and preprocessing data...")
df = df.dropna(subset=['Purchase Amount (USD)'])
df['Review Rating'] = df['Review Rating'].fillna(df['Review Rating'].median())
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'])
df = df.sort_values('Date Purchase')

# Aggregate daily & add better features
df_daily = df.groupby('Date Purchase').agg({
    'Purchase Amount (USD)': 'mean',
    'Item Purchased': 'count',
    'Review Rating': 'mean',
    'Payment Method': lambda x: (x == 'Credit Card').mean()
}).rename(columns={'Payment Method': 'credit_card_ratio'})

# Add more sophisticated time features
df_daily['day_of_week'] = df_daily.index.dayofweek
df_daily['month'] = df_daily.index.month
df_daily['day_of_month'] = df_daily.index.day
df_daily['week_of_year'] = df_daily.index.isocalendar().week
df_daily['quarter'] = df_daily.index.quarter
df_daily['is_weekend'] = df_daily.index.dayofweek.isin([5,6]).astype(int)
df_daily['is_month_end'] = df_daily.index.is_month_end.astype(int)

# More advanced moving averages
df_daily['MA3'] = df_daily['Purchase Amount (USD)'].rolling(window=3).mean()
df_daily['MA7'] = df_daily['Purchase Amount (USD)'].rolling(window=7).mean()
df_daily['MA14'] = df_daily['Purchase Amount (USD)'].rolling(window=14).mean()
df_daily['volatility'] = df_daily['Purchase Amount (USD)'].rolling(window=7).std()

# Add more advanced features
df_daily['lag1'] = df_daily['Purchase Amount (USD)'].shift(1)
df_daily['lag7'] = df_daily['Purchase Amount (USD)'].shift(7) 
df_daily['lag30'] = df_daily['Purchase Amount (USD)'].shift(30)
df_daily['ewm7'] = df_daily['Purchase Amount (USD)'].ewm(span=7).mean()
df_daily['ewm30'] = df_daily['Purchase Amount (USD)'].ewm(span=30).mean()

# Fill missing values more intelligently
df_daily = df_daily.fillna(method='bfill').fillna(method='ffill')

# Transform target variable with log1p
df_daily['Purchase Amount (USD)'] = np.log1p(df_daily['Purchase Amount (USD)'])

# Scale target variable separately
target_scaler = StandardScaler()
df_daily['Purchase Amount (USD)'] = target_scaler.fit_transform(df_daily[['Purchase Amount (USD)']])

# Enhanced exogenous variables
exog_columns = [
    'Item Purchased', 'Review Rating', 'credit_card_ratio',
    'day_of_week', 'day_of_month', 'week_of_year', 
    'month', 'quarter', 'is_weekend', 'is_month_end',
    'MA3', 'MA7', 'MA14', 'volatility',
    'lag1', 'lag7', 'lag30', 'ewm7', 'ewm30'
]

# More robust data transformation
df_daily = df_daily.dropna()

# Scale exogenous features
scaler = StandardScaler()
df_daily[exog_columns] = scaler.fit_transform(df_daily[exog_columns])

# ========== TRAIN-TEST SPLIT (80/20) ==========
print("\nSplitting data 80/20...")
n = len(df_daily)
train_size = int(n * 0.8)
train_data = df_daily.iloc[:train_size]
test_data = df_daily.iloc[train_size:]

# Visualize original vs preprocessed data
plt.figure(figsize=(15,10))

# Plot 1: Original Purchase Amount Distribution
plt.subplot(2,1,1)
df['Purchase Amount (USD)'].hist(bins=50, color='blue', alpha=0.7)
plt.title('Original Purchase Amount Distribution')
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Plot 2: Preprocessed Daily Average Purchase Amount
plt.subplot(2,1,2)
df_daily['Purchase Amount (USD)'].hist(bins=50, color='green', alpha=0.7)
plt.title('Preprocessed Daily Average Purchase Amount')
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize train-test split
plt.figure(figsize=(15,7))
plt.plot(train_data.index, train_data['Purchase Amount (USD)'], 
         label='Training Data', color='blue', alpha=0.7)
plt.plot(test_data.index, test_data['Purchase Amount (USD)'], 
         label='Test Data', color='red', alpha=0.7)
plt.title('Train-Test Split Visualization')
plt.xlabel('Date')
plt.ylabel('Purchase Amount (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

train_exog = train_data[exog_columns]
test_exog = test_data[exog_columns]

# ========== SARIMAX MODEL ==========
print("\nTraining optimized SARIMAX model...")
# Better parameter combinations for SARIMAX
param_combinations = [
    ((3,1,3), (1,1,1,7)),
    ((2,1,2), (2,1,2,7)), 
    ((3,1,2), (1,1,1,14)),
    ((4,1,1), (0,1,1,7)),
    ((2,0,2), (1,1,1,7)),
    ((1,1,1), (0,1,1,7))
]

best_aic = float('inf')
best_model = None
best_params = None

for order, seasonal_order in param_combinations:
    try:
        model = SARIMAX(
            train_data['Purchase Amount (USD)'],
            exog=train_exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit(disp=False, maxiter=200)
        if result.aic < best_aic:
            best_aic = result.aic
            best_model = result
            best_params = (order, seasonal_order)
    except:
        continue

print(f"Best parameters: {best_params}")

# Create future dates for 7-day forecast
future_dates = pd.date_range(
    start=test_data.index[-1] + pd.Timedelta(days=1),
    periods=7,
    freq='D'
)

# Prepare future exogenous features
future_exog = pd.DataFrame(index=future_dates, columns=exog_columns)
future_exog['day_of_week'] = future_dates.dayofweek
future_exog['month'] = future_dates.month
future_exog['day_of_month'] = future_dates.day
future_exog['week_of_year'] = future_dates.isocalendar().week
future_exog['quarter'] = future_dates.quarter
future_exog['is_weekend'] = future_dates.dayofweek.isin([5,6]).astype(int)
future_exog['is_month_end'] = future_dates.is_month_end.astype(int)

# Use last known values for other features
for col in ['Item Purchased', 'Review Rating', 'credit_card_ratio', 
            'MA3', 'MA7', 'MA14', 'volatility', 
            'lag1', 'lag7', 'lag30', 'ewm7', 'ewm30']:
    future_exog[col] = df_daily[col].iloc[-1]

# Scale future exogenous features using same scaler
future_exog = pd.DataFrame(
    scaler.transform(future_exog),
    index=future_dates,
    columns=exog_columns
)

# Generate predictions and inverse transform
pred = best_model.get_forecast(steps=len(test_data), exog=test_exog)
sarimax_pred = target_scaler.inverse_transform(pred.predicted_mean.values.reshape(-1,1))
actual_test = target_scaler.inverse_transform(test_data['Purchase Amount (USD)'].values.reshape(-1,1))

# Generate future predictions and inverse transform 
future_pred = best_model.get_forecast(steps=7, exog=future_exog)
future_values = target_scaler.inverse_transform(future_pred.predicted_mean.values.reshape(-1,1))

# Convert back from log scale
sarimax_pred = np.expm1(sarimax_pred)
actual_test = np.expm1(actual_test) 
future_values = np.expm1(future_values)

# Prevent negative predictions
future_values = np.maximum(0, future_values)
sarimax_pred = np.maximum(0, sarimax_pred)

# Calculate metrics
mae_sarimax = mean_absolute_error(actual_test, sarimax_pred)
rmse = np.sqrt(mean_squared_error(actual_test, sarimax_pred))
r2 = r2_score(actual_test, sarimax_pred)
mape = mean_absolute_percentage_error(actual_test, sarimax_pred) * 100

# ========== VISUALIZATION ==========
plt.figure(figsize=(15,7))
plt.plot(test_data.index, np.maximum(0, actual_test), label='Actual', color='black')
plt.plot(test_data.index, np.maximum(0, sarimax_pred), label='Predicted', color='blue', alpha=0.8)
plt.plot(future_dates, future_values, label='Future Forecast', color='red', linestyle='--')
plt.title('SARIMAX: Actual vs Predicted Values with 7-Day Forecast')
plt.ylabel('Purchase Amount (USD)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== VISUALISASI METRIK PERFORMA SARIMAX ==========
print("\nVisualizing SARIMAX performance metrics...")
plt.figure(figsize=(10, 6))
metrics = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
sarimax_values = [rmse, mae_sarimax, mape, r2*100]
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
print(f"RMSE: {int(rmse)}")  # Rounded to integer
print(f"MAE: {int(mae_sarimax)}")  # Rounded to integer
print(f"MAPE: {int(mape)}%")  # Rounded to integer
print(f"R²: {r2:.2f}")  # Keep 2 decimal places for R²
print(f"{'='*50}")

# Display 7-day forecast
print("\n7-DAY FORECAST:")
print("="*50)
for date, pred in zip(future_dates, future_values):
    print(f"{date.strftime('%Y-%m-%d')}: ${pred[0]:.2f}")  # Access first element of pred array
print("="*50)

# Display last 37 days prediction details
print("\nLAST 37 DAYS PREDICTION DETAILS:")
print("         Date       Actual    Predicted  Difference")
print("="*60)
last_37_actual = actual_test[-37:]
last_37_pred = sarimax_pred[-37:]
last_37_dates = test_data.index[-37:]

for date, act, pred in zip(last_37_dates, last_37_actual, last_37_pred):
    diff = pred[0] - act[0]  # Access first elements
    print(f"{date.strftime('%Y-%m-%d')}  {act[0]:10.2f}  {pred[0]:10.2f}  {diff:10.2f}")

# Optional: Save the model and results
import pickle
print("\nSaving SARIMAX model and results...")
with open('sarimax_model_results.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'params': best_params,
        'predictions': sarimax_pred,
        'actual': actual_test,
        'metrics': {
            'rmse': rmse,
            'mae': mae_sarimax,
            'mape': mape,
            'r2': r2
        }
    }, f)
print("SARIMAX model and results saved to 'sarimax_model_results.pkl'")