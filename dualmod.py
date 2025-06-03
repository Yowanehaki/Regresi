# ========== IMPORT LIBRARIES ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ========== LOAD AND EXPLORE DATA ==========
print("="*80)
print("LOADING AND EXPLORING DATA")
print("="*80)
df = pd.read_csv('Fashion_Retail_Sales.csv')
print("\nSample data:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nData description:")
print(df.describe(include="all"))

# ========== COMMON DATA CLEANING ==========
print("\n" + "="*80)
print("CLEANING DATA")
print("="*80)
df = df.dropna(subset=['Purchase Amount (USD)'])
df['Review Rating'] = df['Review Rating'].fillna(df['Review Rating'].mean())
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'])
df = df.sort_values('Date Purchase')

# Make copies for each model
df_gru = df.copy()
df_sarimax = df.copy()

# ========== GRU MODEL PIPELINE ==========
print("\n" + "="*80)
print("GRU MODEL PIPELINE")
print("="*80)

# --- GRU Preprocessing ---
print("\nPreprocessing data for GRU...")
df_daily_gru = df_gru.set_index('Date Purchase').resample('D').agg({
    'Purchase Amount (USD)': 'sum',
    'Item Purchased': 'count',
    'Review Rating': 'mean'
}).fillna(0)

# Smoothing
df_daily_gru['Purchase Amount (USD)'] = df_daily_gru['Purchase Amount (USD)'].rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')

# Time-based features
df_daily_gru['day_of_week'] = df_daily_gru.index.dayofweek
df_daily_gru['month'] = df_daily_gru.index.month
df_daily_gru['quarter'] = df_daily_gru.index.quarter
df_daily_gru['is_weekend'] = df_daily_gru['day_of_week'].isin([5,6]).astype(int)
df_daily_gru['MA7'] = df_daily_gru['Purchase Amount (USD)'].rolling(window=7).mean()
df_daily_gru['MA30'] = df_daily_gru['Purchase Amount (USD)'].rolling(window=30).mean()
df_daily_gru = df_daily_gru.fillna(method='bfill').fillna(method='ffill')

# Define feature columns for GRU
exog_columns_gru = ['Item Purchased', 'Review Rating', 'day_of_week', 'month', 'quarter', 'is_weekend', 'MA7', 'MA30']

# Scaling features for GRU
exog_scaler_gru = MinMaxScaler()
df_daily_gru[exog_columns_gru] = exog_scaler_gru.fit_transform(df_daily_gru[exog_columns_gru])

# Store original values for later reference
df_daily_gru['Purchase Amount (USD)_original'] = df_daily_gru['Purchase Amount (USD)'].copy()

# Scaling target for GRU
scaler_gru = MinMaxScaler()
scaled_purchase_gru = scaler_gru.fit_transform(df_daily_gru[['Purchase Amount (USD)']])
df_daily_gru['Purchase Amount (USD)'] = scaled_purchase_gru

# --- GRU Train-Test Split ---
print("\nSplitting data 80/20 for GRU model...")
n_gru = len(df_daily_gru)
train_size_gru = int(n_gru * 0.8)
train_data_gru = df_daily_gru.iloc[:train_size_gru]
test_data_gru = df_daily_gru.iloc[train_size_gru:]

# --- GRU Model Preparation ---
print("\nPreparing data sequences for GRU...")
# For GRU: Combine target and exog features as input
gru_features = ['Purchase Amount (USD)'] + exog_columns_gru
gru_data = df_daily_gru[gru_features].values

# Sequence length for GRU
n_steps = 14
print(f"Using sequence length of {n_steps} for GRU")

# Create sequences
X_gru, y_gru = [], []
for i in range(n_steps, len(gru_data)):
    X_gru.append(gru_data[i-n_steps:i])
    y_gru.append(gru_data[i, 0])  # target is the first column
X_gru, y_gru = np.array(X_gru), np.array(y_gru)

# Split 80/20
split_gru = int(len(X_gru) * 0.8)
X_train_gru, y_train_gru = X_gru[:split_gru], y_gru[:split_gru]
X_test_gru, y_test_gru = X_gru[split_gru:], y_gru[split_gru:]

print(f"GRU training data: {X_train_gru.shape}")
print(f"GRU testing data: {X_test_gru.shape}")

# --- Build and Train GRU Model ---
print("\nBuilding and training GRU model...")
gru_model = Sequential([
    # Multi-layer GRU with regularization
    GRU(128, return_sequences=True, input_shape=(n_steps, X_gru.shape[2]), 
        recurrent_dropout=0.2),
    GRU(64, return_sequences=True, recurrent_dropout=0.2),
    GRU(32, return_sequences=False, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  # Linear activation for regression
])

# Use a lower learning rate
optimizer = Adam(learning_rate=0.0005)
gru_model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])  # Huber loss is more robust
gru_model.summary()

# Callback functions for GRU
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    min_delta=0.00001,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=15,
    min_lr=1e-7,
    verbose=1
)

# Train GRU model
history_gru = gru_model.fit(
    X_train_gru, y_train_gru,
    epochs=300,
    batch_size=32,
    validation_data=(X_test_gru, y_test_gru),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# --- GRU Predictions and Evaluation ---
print("\nGenerating GRU predictions and evaluating model...")
y_pred_gru = gru_model.predict(X_test_gru)
y_pred_gru_inv = scaler_gru.inverse_transform(y_pred_gru)
y_test_gru_inv = scaler_gru.inverse_transform(y_test_gru.reshape(-1, 1))

# GRU Future Predictions (7 days)
print("\nPredicting next 7 days with GRU...")
last_sequence_gru = X_test_gru[-1:]
future_predictions_gru = []

for _ in range(7):
    # Get prediction for next day
    next_pred = gru_model.predict(last_sequence_gru)
    future_predictions_gru.append(next_pred[0, 0])
    
    # Update the sequence for next prediction
    new_seq = last_sequence_gru[0][1:].copy()
    new_row = last_sequence_gru[0][-1].copy()
    new_row[0] = next_pred[0, 0]  # Update only the target value
    last_sequence_gru = np.append(new_seq, [new_row], axis=0).reshape(1, n_steps, X_gru.shape[2])

# Inverse transform predictions
future_pred_inv_gru = scaler_gru.inverse_transform(np.array(future_predictions_gru).reshape(-1, 1))

# Create future dates for GRU
future_dates_gru = pd.date_range(start=test_data_gru.index[-1], periods=8)[1:]

# GRU Metrics
mae_gru = mean_absolute_error(y_test_gru_inv, y_pred_gru_inv)
mape_gru = mean_absolute_percentage_error(y_test_gru_inv, y_pred_gru_inv) * 100
r2_gru = r2_score(y_test_gru_inv, y_pred_gru_inv)
rmse_gru = np.sqrt(mean_squared_error(y_test_gru_inv, y_pred_gru_inv))

print("\n=== GRU Evaluation ===")
print(f"RMSE  : {rmse_gru:.2f}")
print(f"MAE   : {mae_gru:.2f}")
print(f"MAPE  : {mape_gru:.2f}%")
print(f"R²    : {r2_gru:.4f}")

print("\nGRU Predicted values for next 7 days:")
for date, pred in zip(future_dates_gru, future_pred_inv_gru):
    print(f"{date.date()}: ${pred[0]:.2f}")

# Skip visualization for now - will do it after SARIMAX predictions
# Continue with SARIMAX pipeline

# ========== SARIMAX MODEL PIPELINE ==========
print("\n" + "="*80)
print("SARIMAX MODEL PIPELINE")
print("="*80)

# --- SARIMAX Preprocessing ---
print("\nPreprocessing data for SARIMAX...")
# Aggregate daily & add better features
df_daily_sarimax = df_sarimax.groupby('Date Purchase').agg({
    'Purchase Amount (USD)': 'mean',
    'Item Purchased': 'count',
    'Review Rating': 'mean',
    'Payment Method': lambda x: (x == 'Credit Card').mean()
}).rename(columns={'Payment Method': 'credit_card_ratio'})

# Add more sophisticated time features
df_daily_sarimax['day_of_week'] = df_daily_sarimax.index.dayofweek
df_daily_sarimax['month'] = df_daily_sarimax.index.month
df_daily_sarimax['day_of_month'] = df_daily_sarimax.index.day
df_daily_sarimax['week_of_year'] = df_daily_sarimax.index.isocalendar().week
df_daily_sarimax['quarter'] = df_daily_sarimax.index.quarter
df_daily_sarimax['is_weekend'] = df_daily_sarimax.index.dayofweek.isin([5,6]).astype(int)
df_daily_sarimax['is_month_end'] = df_daily_sarimax.index.is_month_end.astype(int)

# More advanced moving averages
df_daily_sarimax['MA3'] = df_daily_sarimax['Purchase Amount (USD)'].rolling(window=3).mean()
df_daily_sarimax['MA7'] = df_daily_sarimax['Purchase Amount (USD)'].rolling(window=7).mean()
df_daily_sarimax['MA14'] = df_daily_sarimax['Purchase Amount (USD)'].rolling(window=14).mean()
df_daily_sarimax['volatility'] = df_daily_sarimax['Purchase Amount (USD)'].rolling(window=7).std()

# Add more advanced features
df_daily_sarimax['lag1'] = df_daily_sarimax['Purchase Amount (USD)'].shift(1)
df_daily_sarimax['lag7'] = df_daily_sarimax['Purchase Amount (USD)'].shift(7) 
df_daily_sarimax['lag30'] = df_daily_sarimax['Purchase Amount (USD)'].shift(30)
df_daily_sarimax['ewm7'] = df_daily_sarimax['Purchase Amount (USD)'].ewm(span=7).mean()
df_daily_sarimax['ewm30'] = df_daily_sarimax['Purchase Amount (USD)'].ewm(span=30).mean()

# Store original values before transformation
df_daily_sarimax['Purchase Amount (USD)_original'] = df_daily_sarimax['Purchase Amount (USD)'].copy()

# Fill missing values more intelligently
df_daily_sarimax = df_daily_sarimax.fillna(method='bfill').fillna(method='ffill')

# Transform target variable with log1p
df_daily_sarimax['Purchase Amount (USD)'] = np.log1p(df_daily_sarimax['Purchase Amount (USD)'])

# Scale target variable separately
target_scaler_sarimax = StandardScaler()
df_daily_sarimax['Purchase Amount (USD)'] = target_scaler_sarimax.fit_transform(df_daily_sarimax[['Purchase Amount (USD)']])

# Enhanced exogenous variables
exog_columns_sarimax = [
    'Item Purchased', 'Review Rating', 'credit_card_ratio',
    'day_of_week', 'day_of_month', 'week_of_year', 
    'month', 'quarter', 'is_weekend', 'is_month_end',
    'MA3', 'MA7', 'MA14', 'volatility',
    'lag1', 'lag7', 'lag30', 'ewm7', 'ewm30'
]

# More robust data transformation
df_daily_sarimax = df_daily_sarimax.dropna()

# Scale exogenous features
scaler_sarimax = StandardScaler()
df_daily_sarimax[exog_columns_sarimax] = scaler_sarimax.fit_transform(df_daily_sarimax[exog_columns_sarimax])

# --- SARIMAX Train-Test Split ---
print("\nSplitting data 80/20 for SARIMAX model...")
n_sarimax = len(df_daily_sarimax)
train_size_sarimax = int(n_sarimax * 0.8)
train_data_sarimax = df_daily_sarimax.iloc[:train_size_sarimax]
test_data_sarimax = df_daily_sarimax.iloc[train_size_sarimax:]

# Prepare exogenous data for SARIMAX
train_exog_sarimax = train_data_sarimax[exog_columns_sarimax]
test_exog_sarimax = test_data_sarimax[exog_columns_sarimax]

# --- Train SARIMAX Model ---
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
best_sarimax_model = None
best_params = None

for order, seasonal_order in param_combinations:
    try:
        model = SARIMAX(
            train_data_sarimax['Purchase Amount (USD)'],
            exog=train_exog_sarimax,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit(disp=False, maxiter=200)
        if result.aic < best_aic:
            best_aic = result.aic
            best_sarimax_model = result
            best_params = (order, seasonal_order)
            print(f"New best model found: {best_params} with AIC: {best_aic:.2f}")
    except Exception as e:
        print(f"Error with parameters {order}, {seasonal_order}: {str(e)}")
        continue

print(f"\nBest SARIMAX parameters: {best_params} with AIC: {best_aic:.2f}")

# --- SARIMAX Predictions and Evaluation ---
print("\nGenerating SARIMAX predictions...")
pred_sarimax = best_sarimax_model.get_forecast(steps=len(test_data_sarimax), exog=test_exog_sarimax)
sarimax_pred = target_scaler_sarimax.inverse_transform(pred_sarimax.predicted_mean.values.reshape(-1,1))
actual_test_sarimax = target_scaler_sarimax.inverse_transform(test_data_sarimax['Purchase Amount (USD)'].values.reshape(-1,1))

# Convert back from log scale for SARIMAX
sarimax_pred = np.expm1(sarimax_pred)
actual_test_sarimax = np.expm1(actual_test_sarimax)

# Prevent negative predictions
sarimax_pred = np.maximum(0, sarimax_pred)

# SARIMAX Future Predictions (7 days)
print("\nPredicting next 7 days with SARIMAX...")
# Create future dates for 7-day forecast
future_dates_sarimax = pd.date_range(
    start=test_data_sarimax.index[-1] + pd.Timedelta(days=1),
    periods=7,
    freq='D'
)

# Prepare future exogenous features
future_exog_sarimax = pd.DataFrame(index=future_dates_sarimax, columns=exog_columns_sarimax)
future_exog_sarimax['day_of_week'] = future_dates_sarimax.dayofweek
future_exog_sarimax['month'] = future_dates_sarimax.month
future_exog_sarimax['day_of_month'] = future_dates_sarimax.day
future_exog_sarimax['week_of_year'] = future_dates_sarimax.isocalendar().week
future_exog_sarimax['quarter'] = future_dates_sarimax.quarter
future_exog_sarimax['is_weekend'] = future_dates_sarimax.dayofweek.isin([5,6]).astype(int)
future_exog_sarimax['is_month_end'] = future_dates_sarimax.is_month_end.astype(int)

# Use last known values for other features
for col in ['Item Purchased', 'Review Rating', 'credit_card_ratio', 
            'MA3', 'MA7', 'MA14', 'volatility', 
            'lag1', 'lag7', 'lag30', 'ewm7', 'ewm30']:
    future_exog_sarimax[col] = df_daily_sarimax[col].iloc[-1]

# Scale future exogenous features using same scaler
future_exog_sarimax = pd.DataFrame(
    scaler_sarimax.transform(future_exog_sarimax),
    index=future_dates_sarimax,
    columns=exog_columns_sarimax
)

# Generate future predictions and inverse transform 
future_pred_sarimax = best_sarimax_model.get_forecast(steps=7, exog=future_exog_sarimax)
future_values_sarimax = target_scaler_sarimax.inverse_transform(future_pred_sarimax.predicted_mean.values.reshape(-1,1))
future_values_sarimax = np.expm1(future_values_sarimax)
future_values_sarimax = np.maximum(0, future_values_sarimax)

# SARIMAX Metrics
mae_sarimax = mean_absolute_error(actual_test_sarimax, sarimax_pred)
mape_sarimax = mean_absolute_percentage_error(actual_test_sarimax, sarimax_pred) * 100
r2_sarimax = r2_score(actual_test_sarimax, sarimax_pred)
rmse_sarimax = np.sqrt(mean_squared_error(actual_test_sarimax, sarimax_pred))

print("\n=== SARIMAX Evaluation ===")
print(f"Best parameters: {best_params}")
print(f"RMSE  : {rmse_sarimax:.2f}")
print(f"MAE   : {mae_sarimax:.2f}")
print(f"MAPE  : {mape_sarimax:.2f}%")
print(f"R²    : {r2_sarimax:.4f}")

print("\nSARIMAX Predicted values for next 7 days:")
for date, pred in zip(future_dates_sarimax, future_values_sarimax):
    print(f"{date.date()}: ${pred[0]:.2f}")

# --- GRU Visualization ---
plt.figure(figsize=(15,7))
aligned_test_dates_gru = test_data_gru.index[-len(y_test_gru_inv):]

# Get min and max values across both models for consistent y-axis
y_min = min(np.min(y_test_gru_inv), np.min(actual_test_sarimax))
y_max = max(np.max(y_test_gru_inv), np.max(actual_test_sarimax))

# Plot GRU results with consistent y-axis
plt.plot(aligned_test_dates_gru, np.maximum(0, y_test_gru_inv), label='Actual Values', color=colors[0], linewidth=2)
plt.plot(aligned_test_dates_gru, np.maximum(0, y_pred_gru_inv), label='GRU Predictions', color=colors[1], alpha=0.8, linewidth=2)
plt.plot(future_dates_gru, future_pred_inv_gru, label='GRU Future Forecast', color=colors[2], linestyle='--', linewidth=2)
plt.title('GRU Model: Sales Forecasting Results', fontsize=16)
plt.ylabel('Purchase Amount (USD)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(y_min, y_max * 1.1)  # Consistent y-axis with 10% padding
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('gru_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# --- SARIMAX Visualization ---
plt.figure(figsize=(15,7))

# Plot SARIMAX results with same y-axis limits
plt.plot(test_data_sarimax.index, np.maximum(0, actual_test_sarimax), label='Actual Values', color=colors[0], linewidth=2)
plt.plot(test_data_sarimax.index, np.maximum(0, sarimax_pred), label='SARIMAX Predictions', color=colors[3], alpha=0.8, linewidth=2)
plt.plot(future_dates_sarimax, future_values_sarimax, label='SARIMAX Future Forecast', color=colors[4], linestyle='--', linewidth=2)
plt.title('SARIMAX Model: Sales Forecasting Results', fontsize=16)
plt.ylabel('Purchase Amount (USD)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(y_min, y_max * 1.1)  # Using same y-axis limits as GRU plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sarimax_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== MODEL COMPARISON AND ENSEMBLE ==========
print("\n" + "="*80)
print("MODEL COMPARISON AND ENSEMBLE FORECASTING")
print("="*80)

# --- Performance Metrics Comparison ---
print("\nComparing model performance metrics...")

# Create a DataFrame for metrics comparison
metrics_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'R² Score'],
    'GRU': [rmse_gru, mae_gru, mape_gru, r2_gru],
    'SARIMAX': [rmse_sarimax, mae_sarimax, mape_sarimax, r2_sarimax]
})

print(metrics_df)

# Visualization of metrics comparison with enhanced styling
plt.figure(figsize=(12, 8))
metrics = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
gru_values = [rmse_gru, mae_gru, mape_gru, r2_gru*100]
sarimax_values = [rmse_sarimax, mae_sarimax, mape_sarimax, r2_sarimax*100]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, gru_values, width, label='GRU', color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
rects2 = ax.bar(x + width/2, sarimax_values, width, label='SARIMAX', color=colors[3], alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels on top of bars
def autolabel(rects, is_r2=False):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if is_r2[i]:
            ax.annotate(f'{height/100:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold')
        else:
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold')

is_r2 = [False, False, False, True]
autolabel(rects1, is_r2)
autolabel(rects2, is_r2)

ax.set_title('Model Performance Comparison', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Metrics', fontsize=14, labelpad=10)
ax.set_ylabel('Values', fontsize=14, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=14, frameon=True, facecolor='white', edgecolor='gray')

# Add a note about metrics interpretation
fig.text(0.5, 0.01,
         'Note: For RMSE, MAE, MAPE → lower is better | For R² → higher is better',
         ha='center', fontsize=11, style='italic')

# Add grid lines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Set the background color
ax.set_facecolor('#f8f9fa')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 7-Day Forecast Comparison ---
# Align dates from both models' forecasts
print("\nComparing 7-day forecasts from both models...")
forecast_dates = pd.date_range(
    start=min(future_dates_gru[0], future_dates_sarimax[0]),
    periods=7,
    freq='D'
)

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'GRU Forecast': [pred[0] for pred in future_pred_inv_gru],
    'SARIMAX Forecast': [pred[0] for pred in future_values_sarimax],
})

# Add ensemble forecast (average of both models)
forecast_df['Ensemble Forecast'] = (forecast_df['GRU Forecast'] + forecast_df['SARIMAX Forecast']) / 2

# Display forecast comparison
print("\n--- 7-DAY FORECAST COMPARISON ---")
print(forecast_df.to_string(index=False, float_format='${:.2f}'.format))

# Calculate the difference and percentage difference between models
forecast_df['Difference'] = forecast_df['GRU Forecast'] - forecast_df['SARIMAX Forecast']
forecast_df['Percent Difference (%)'] = (forecast_df['Difference'] / 
                                         ((forecast_df['GRU Forecast'] + forecast_df['SARIMAX Forecast']) / 2)) * 100

print("\n--- MODEL FORECAST DIFFERENCES ---")
# Simplify the formatting to avoid the error
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
print("\nForecast Comparison:")
comparison_df = forecast_df[['Date', 'GRU Forecast', 'SARIMAX Forecast']]
print(comparison_df.to_string(index=False))

print("\nModel Differences:")
diff_df = forecast_df[['Date', 'Difference', 'Percent Difference (%)']]
print(diff_df.to_string(index=False))

# Reset float format to default
pd.reset_option('display.float_format')

# Plot enhanced forecast comparison
plt.figure(figsize=(16, 10))

# Create line plot with markers for clarity
plt.plot(forecast_df['Date'], forecast_df['GRU Forecast'], 'o-', 
         label='GRU Forecast', color=colors[1], linewidth=3, markersize=10)
plt.plot(forecast_df['Date'], forecast_df['SARIMAX Forecast'], 's-', 
         label='SARIMAX Forecast', color=colors[3], linewidth=3, markersize=10)
plt.plot(forecast_df['Date'], forecast_df['Ensemble Forecast'], 'd-', 
         label='Ensemble Forecast', color='red', linewidth=4, markersize=12)

# Add value annotations
for i, row in forecast_df.iterrows():
    # GRU annotation
    plt.annotate(f"${row['GRU Forecast']:.2f}", 
                xy=(row['Date'], row['GRU Forecast']),
                xytext=(0, 15),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=colors[1])
    
    # SARIMAX annotation
    plt.annotate(f"${row['SARIMAX Forecast']:.2f}", 
                xy=(row['Date'], row['SARIMAX Forecast']),
                xytext=(0, -20),
                textcoords="offset points",
                ha='center', va='top',
                fontsize=10, fontweight='bold', color=colors[3])
    
    # Ensemble annotation
    plt.annotate(f"${row['Ensemble Forecast']:.2f}", 
                xy=(row['Date'], row['Ensemble Forecast']),
                xytext=(0, 15),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='red')

# Add title and labels with enhanced styling
plt.title('7-Day Forecast Comparison: GRU vs SARIMAX vs Ensemble', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=16, labelpad=15)
plt.ylabel('Purchase Amount (USD)', fontsize=16, labelpad=15)

# Improve grid and background
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().set_facecolor('#f8f9fa')

# Enhance legend
plt.legend(fontsize=16, loc='upper left', frameon=True, facecolor='white', edgecolor='gray')

# Add text box explaining ensemble
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
plt.figtext(0.5, 0.02, 
           "Note: Ensemble Forecast is the average of GRU and SARIMAX predictions,\n"
           "potentially offering more robust forecasting by leveraging strengths of both approaches.",
           ha='center', fontsize=12, bbox=props)

plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig('forecast_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Determine which model performs better ---
better_model = "GRU" if r2_gru > r2_sarimax else "SARIMAX"
print(f"\nBased on R² score, the {better_model} model performs better on this dataset.")
print(f"GRU R² score: {r2_gru:.4f}")
print(f"SARIMAX R² score: {r2_sarimax:.4f}")

# --- Visualization of All Predictions on One Plot ---
# Find common date range for fair comparison
common_start = max(aligned_test_dates_gru[0], test_data_sarimax.index[0])
common_end = min(aligned_test_dates_gru[-1], test_data_sarimax.index[-1])

# Filter data for common date range
gru_idx_start = np.where(aligned_test_dates_gru >= common_start)[0][0]
gru_idx_end = np.where(aligned_test_dates_gru <= common_end)[0][-1] + 1
sarimax_idx_start = np.where(test_data_sarimax.index >= common_start)[0][0]
sarimax_idx_end = np.where(test_data_sarimax.index <= common_end)[0][-1] + 1

# Create comprehensive comparison plot
plt.figure(figsize=(18, 10))

# Plot actual values
plt.plot(aligned_test_dates_gru[gru_idx_start:gru_idx_end], 
         np.maximum(0, y_test_gru_inv[gru_idx_start:gru_idx_end]), 
         label='Actual', color='black', linewidth=3)

# Plot GRU predictions
plt.plot(aligned_test_dates_gru[gru_idx_start:gru_idx_end], 
         np.maximum(0, y_pred_gru_inv[gru_idx_start:gru_idx_end]), 
         label='GRU Predictions', color=colors[1], linewidth=2, alpha=0.8)

# Plot SARIMAX predictions
plt.plot(test_data_sarimax.index[sarimax_idx_start:sarimax_idx_end], 
         np.maximum(0, sarimax_pred[sarimax_idx_start:sarimax_idx_end]), 
         label='SARIMAX Predictions', color=colors[3], linewidth=2, alpha=0.8)

# Plot future predictions for both models
plt.plot(future_dates_gru, future_pred_inv_gru, 
         label='GRU Future Forecast', color=colors[1], linestyle='--', linewidth=2)
plt.plot(future_dates_sarimax, future_values_sarimax, 
         label='SARIMAX Future Forecast', color=colors[3], linestyle='--', linewidth=2)

# Calculate and plot ensemble for future predictions
# Ensure the dates are aligned first
future_dates = pd.date_range(
    start=min(future_dates_gru[0], future_dates_sarimax[0]),
    periods=7,
    freq='D'
)

ensemble_future = []
for i in range(7):
    gru_val = future_pred_inv_gru[i][0]
    sarimax_val = future_values_sarimax[i][0]
    ensemble_future.append((gru_val + sarimax_val) / 2)

plt.plot(future_dates, ensemble_future, 
         label='Ensemble Future Forecast', color='red', linestyle='--', linewidth=3)

# Add vertical line separating historical data from forecasts
plt.axvline(x=max(aligned_test_dates_gru[-1], test_data_sarimax.index[-1]), 
            color='gray', linestyle='-', linewidth=2, alpha=0.7)
plt.text(max(aligned_test_dates_gru[-1], test_data_sarimax.index[-1]), 
         plt.ylim()[1]*0.9, ' Future Forecast →', 
         fontsize=14, color='gray', ha='left')

# Add styling and information
plt.title('Comprehensive Model Comparison: GRU vs SARIMAX with Future Forecasts', 
          fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=16, labelpad=15)
plt.ylabel('Purchase Amount (USD)', fontsize=16, labelpad=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().set_facecolor('#f8f9fa')
plt.legend(fontsize=14, loc='upper left', frameon=True, facecolor='white', edgecolor='gray')

# Add annotation with model performance metrics
metrics_text = (f"GRU: RMSE={rmse_gru:.2f}, R²={r2_gru:.4f}\n"
                f"SARIMAX: RMSE={rmse_sarimax:.2f}, R²={r2_sarimax:.4f}")
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
plt.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction', 
             fontsize=12, ha='left', va='top', bbox=props)

plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Save models
print("\nSaving models...")
# Save GRU model
gru_model.save('gru_model.h5')
print("GRU model saved to 'gru_model.h5'")

# Save SARIMAX model
with open('sarimax_model_results.pkl', 'wb') as f:
    pickle.dump({
        'model': best_sarimax_model,
        'params': best_params,
        'predictions': sarimax_pred,
        'actual': actual_test_sarimax,
        'metrics': {
            'rmse': rmse_sarimax,
            'mae': mae_sarimax,
            'mape': mape_sarimax,
            'r2': r2_sarimax
        }
    }, f)
print("SARIMAX model and results saved to 'sarimax_model_results.pkl'")

# ========== CONCLUSION ==========
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print(f"""
This analysis combined two powerful forecasting approaches for fashion retail sales data:

1. GRU (Deep Learning): 
   - R² Score: {r2_gru:.4f}
   - RMSE: {rmse_gru:.2f}
   - Better at capturing complex non-linear patterns
   
2. SARIMAX (Statistical): 
   - R² Score: {r2_sarimax:.4f}
   - RMSE: {rmse_sarimax:.2f}
   - Better interpretability and handles seasonality well
   
Overall, the {better_model} model performed better on this dataset.

The ensemble approach (average of both models) potentially offers more robust forecasting
by leveraging the strengths of both approaches, especially for future predictions.

All visualizations have been saved as high-resolution PNG files.
""")