# ========== IMPORT LIBRARY ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler # Consolidated
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam # Specific to TF/Keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # Specific to TF/Keras
from statsmodels.tsa.statespace.sarimax import SARIMAX # Specific to Statsmodels
import warnings
import pickle # For saving SARIMAX model

warnings.filterwarnings('ignore')

# ========== LOAD DAN EKSPLORASI DATA AWAL ==========
print("="*30 + " LOAD DAN EKSPLORASI DATA AWAL " + "="*30)
print("Loading and exploring data...")
df_original = pd.read_csv('Fashion_Retail_Sales.csv')
print("\nSample data (Original):")
print(df_original.head())
print("\nData info (Original):")
df_original.info() # .info() prints directly
print("\nData description (Original):")
print(df_original.describe(include="all"))

# Plot Original Purchase Amount Distribution (once)
plt.figure(figsize=(10,6))
df_original['Purchase Amount (USD)'].hist(bins=50, color='purple', alpha=0.7)
plt.title('Original Purchase Amount Distribution (Raw Data)')
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ###########################################################################
# # ========== SECTION 1: GRU MODEL ==========
# ###########################################################################
print("\n\n" + "="*30 + " GRU MODEL SECTION " + "="*30)

# ========== DATA CLEANING (GRU) ==========
print("\nCleaning data for GRU...")
df_gru = df_original.copy()
df_gru = df_gru.dropna(subset=['Purchase Amount (USD)'])
df_gru['Review Rating'] = df_gru['Review Rating'].fillna(df_gru['Review Rating'].mean())
df_gru['Date Purchase'] = pd.to_datetime(df_gru['Date Purchase'])
df_gru = df_gru.sort_values('Date Purchase')

# ========== PREPROCESSING DATA (GRU) ==========
print("\nPreprocessing data for GRU...")
df_daily_gru = df_gru.set_index('Date Purchase').resample('D').agg({
    'Purchase Amount (USD)': 'sum',
    'Item Purchased': 'count',
    'Review Rating': 'mean'
}).fillna(0)

# Smoothing for GRU
df_daily_gru['Purchase Amount (USD)'] = df_daily_gru['Purchase Amount (USD)'].rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')

# Time-based features for GRU
df_daily_gru['day_of_week'] = df_daily_gru.index.dayofweek
df_daily_gru['month'] = df_daily_gru.index.month
df_daily_gru['quarter'] = df_daily_gru.index.quarter
df_daily_gru['is_weekend'] = df_daily_gru['day_of_week'].isin([5,6]).astype(int)
df_daily_gru['MA7'] = df_daily_gru['Purchase Amount (USD)'].rolling(window=7).mean()
df_daily_gru['MA30'] = df_daily_gru['Purchase Amount (USD)'].rolling(window=30).mean()
df_daily_gru = df_daily_gru.fillna(method='bfill').fillna(method='ffill')

# Definisi feature columns for GRU
exog_columns_gru = ['Item Purchased', 'Review Rating', 'day_of_week', 'month', 'quarter', 'is_weekend', 'MA7', 'MA30']

# Scaling features for GRU
exog_scaler_gru = MinMaxScaler()
df_daily_gru[exog_columns_gru] = exog_scaler_gru.fit_transform(df_daily_gru[exog_columns_gru])

# Scaling target for GRU
target_scaler_gru = MinMaxScaler()
df_daily_gru['Purchase Amount (USD)'] = target_scaler_gru.fit_transform(df_daily_gru[['Purchase Amount (USD)']])

# Visualize preprocessed data for GRU
plt.figure(figsize=(15,5))
df_daily_gru['Purchase Amount (USD)'].hist(bins=50, color='green', alpha=0.7)
plt.title('Preprocessed Daily Sum Purchase Amount Distribution (GRU)')
plt.xlabel('Scaled Purchase Amount (USD)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,7))
plt.plot(df_daily_gru.index, df_daily_gru['Purchase Amount (USD)'],
         label='Preprocessed Data (GRU)', color='blue', alpha=0.7)
plt.title('Preprocessed Time Series Data (GRU)')
plt.xlabel('Date')
plt.ylabel('Scaled Purchase Amount (Sum)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== TRAIN-TEST SPLIT (GRU - 80/20) ==========
print("\nSplitting GRU data 80/20...")
n_gru = len(df_daily_gru)
train_size_gru = int(n_gru * 0.8)
train_data_gru = df_daily_gru.iloc[:train_size_gru]
test_data_gru = df_daily_gru.iloc[train_size_gru:]

# ========== PERSIAPAN DATA UNTUK GRU ==========
print("\nPreparing data for GRU model...")
gru_features = ['Purchase Amount (USD)'] + exog_columns_gru
gru_data_values = df_daily_gru[gru_features].values

# Sequence length untuk GRU
n_steps_gru = 14
print(f"Using sequence length of {n_steps_gru} for GRU")

# Create sequences for GRU
X_gru, y_gru = [], []
for i in range(n_steps_gru, len(gru_data_values)):
    X_gru.append(gru_data_values[i-n_steps_gru:i])
    y_gru.append(gru_data_values[i, 0]) # target tetap kolom pertama
X_gru, y_gru = np.array(X_gru), np.array(y_gru)

# Split sequences 80/20 for GRU
split_gru = int(len(X_gru) * 0.8)
X_train_gru, y_train_gru = X_gru[:split_gru], y_gru[:split_gru]
X_test_gru, y_test_gru = X_gru[split_gru:], y_gru[split_gru:]

print(f"GRU training data shape: {X_train_gru.shape}")
print(f"GRU testing data shape: {X_test_gru.shape}")

# ========== MEMBANGUN MODEL GRU ==========
print("\nBuilding GRU model...")
model_gru = Sequential([
    GRU(128, return_sequences=True, input_shape=(n_steps_gru, X_gru.shape[2]), recurrent_dropout=0.2),
    GRU(64, return_sequences=True, recurrent_dropout=0.2),
    GRU(32, return_sequences=False, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

optimizer_gru = Adam(learning_rate=0.0005)
model_gru.compile(optimizer=optimizer_gru, loss='huber', metrics=['mae'])
model_gru.summary()

# ========== CALLBACK FUNCTIONS (GRU) ==========
early_stopping_gru = EarlyStopping(
    monitor='val_loss', patience=30, restore_best_weights=True, min_delta=0.00001, verbose=1
)
reduce_lr_gru = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=15, min_lr=1e-7, verbose=1
)

# ========== TRAINING MODEL GRU ==========
print("\nTraining GRU model...")
history_gru = model_gru.fit(
    X_train_gru, y_train_gru,
    epochs=300, batch_size=32,
    validation_data=(X_test_gru, y_test_gru),
    callbacks=[early_stopping_gru, reduce_lr_gru],
    verbose=1
)

# ========== PREDIKSI DAN INVERSE TRANSFORM GRU ==========
print("\nGenerating GRU predictions...")
y_pred_gru_scaled = model_gru.predict(X_test_gru)
y_pred_gru_inv = target_scaler_gru.inverse_transform(y_pred_gru_scaled)
y_test_gru_inv = target_scaler_gru.inverse_transform(y_test_gru.reshape(-1, 1))

# ========== EVALUASI GRU ==========
mae_metric_gru = mean_absolute_error(y_test_gru_inv, y_pred_gru_inv)
mape_metric_gru = mean_absolute_percentage_error(y_test_gru_inv, y_pred_gru_inv) * 100
r2_metric_gru = r2_score(y_test_gru_inv, y_pred_gru_inv)
rmse_metric_gru = np.sqrt(mean_squared_error(y_test_gru_inv, y_pred_gru_inv))

print("\n=== GRU Evaluation ===")
print(f"RMSE  : {rmse_metric_gru:.2f}")
print(f"MAE   : {mae_metric_gru:.2f}")
print(f"MAPE  : {mape_metric_gru:.2f}%")
print(f"R2    : {r2_metric_gru:.4f}")

# ========== PREDIKSI 7 HARI KEDEPAN (GRU) ==========
print("\nPredicting next 7 days with GRU...")
last_sequence_gru = X_test_gru[-1:].copy() # Ensure it's a copy to modify
future_predictions_gru_scaled = []

for _ in range(7):
    next_pred_scaled = model_gru.predict(last_sequence_gru)
    future_predictions_gru_scaled.append(next_pred_scaled[0, 0])
    
    new_seq_gru = last_sequence_gru[0][1:].copy()
    new_row_gru = last_sequence_gru[0][-1].copy()
    new_row_gru[0] = next_pred_scaled[0, 0] # Update only the target value (first feature)
    last_sequence_gru = np.append(new_seq_gru, [new_row_gru], axis=0).reshape(1, n_steps_gru, X_gru.shape[2])

future_pred_gru_inv = target_scaler_gru.inverse_transform(np.array(future_predictions_gru_scaled).reshape(-1, 1))
future_dates_gru = pd.date_range(start=test_data_gru.index[-1] + pd.Timedelta(days=1), periods=7)


print("\nPredicted values for next 7 days (GRU):")
for date, pred in zip(future_dates_gru, future_pred_gru_inv):
    print(f"{date.date()}: ${pred[0]:.2f}")

# ========== DETAIL VISUALISASI (GRU) =========
# Show last 37 days comparison (GRU)
plt.figure(figsize=(15,7))
# Adjust indices for plotting based on actual y_test_gru_inv and y_pred_gru_inv lengths
num_test_samples_gru = len(y_test_gru_inv)
dates_for_test_plot_gru = test_data_gru.index[-num_test_samples_gru:]

last_37_actual_gru = y_test_gru_inv[-37:]
last_37_pred_gru = y_pred_gru_inv[-37:]
last_37_dates_gru = dates_for_test_plot_gru[-37:]

plt.plot(last_37_dates_gru, last_37_actual_gru,
         label='Actual (GRU)', linewidth=2, color='black')
plt.plot(last_37_dates_gru, last_37_pred_gru,
         label='Predicted (GRU)', linewidth=2, color='blue', alpha=0.8)
plt.plot(future_dates_gru, future_pred_gru_inv,
         label='Future Predictions (GRU)', linewidth=2, color='red', linestyle='--')

plt.title('GRU: Last 37 Days of Predictions + 7 Days Future Forecast')
plt.xlabel('Date')
plt.ylabel('Purchase Amount (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Show prediction details for last 37 days (GRU)
print("\nLast 37 days prediction details (GRU):")
comparison_df_gru = pd.DataFrame({
    'Date': last_37_dates_gru,
    'Actual': last_37_actual_gru.flatten(),
    'Predicted': last_37_pred_gru.flatten(),
    'Difference': abs(last_37_actual_gru.flatten() - last_37_pred_gru.flatten())
})
print(comparison_df_gru)

# Save GRU model
model_gru.save('gru_model.h5')
print("\nGRU model saved to 'gru_model.h5'")

# ========== VISUALIZATION (Overall GRU) =========
plt.figure(figsize=(15,7))
plt.plot(dates_for_test_plot_gru, np.maximum(0, y_test_gru_inv), label='Actual (GRU)', color='black')
plt.plot(dates_for_test_plot_gru, np.maximum(0, y_pred_gru_inv), label='Predicted (GRU)', color='blue', alpha=0.8)
plt.plot(future_dates_gru, np.maximum(0, future_pred_gru_inv), label='Future Forecast (GRU)', color='red', linestyle='--')
plt.title('GRU: Actual vs Predicted Values with 7-Day Forecast')
plt.ylabel('Purchase Amount (USD)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== VISUALISASI METRIK PERFORMA GRU =========
print("\nVisualizing GRU performance metrics...")
plt.figure(figsize=(10, 6))
metrics_gru_names = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
metric_values_gru = [rmse_metric_gru, mae_metric_gru, mape_metric_gru, r2_metric_gru*100] # R2 is multiplied by 100 for percentage view if desired
x_gru = np.arange(len(metrics_gru_names))
width_gru = 0.5
bars_gru = plt.bar(x_gru, metric_values_gru, width_gru, label='GRU', color='blue', alpha=0.7)

for i, bar in enumerate(bars_gru):
    if i == 3:  # R² Score
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(metric_values_gru), # Adjust offset based on max value
                 f'{metric_values_gru[i]/100:.3f}', ha='center', va='bottom', fontweight='bold') # Display as decimal for R2
    else:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(metric_values_gru),
                 f'{metric_values_gru[i]:.1f}', ha='center', va='bottom', fontweight='bold')

plt.title('GRU Model Performance Metrics', fontsize=16, fontweight='bold')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks(x_gru, metrics_gru_names)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, max(metric_values_gru) * 1.1) # Adjust y-limit for better text visibility
plt.figtext(0.5, 0.01, # Adjusted y position for figtext
            'Note: For RMSE, MAE, MAPE → lower is better | For R² → higher is better',
            ha='center', fontsize=10, style='italic')
plt.tight_layout(rect=[0, 0.03, 1, 1]) # Adjust layout to make space for figtext
plt.show()


# ###########################################################################
# # ========== SECTION 2: SARIMAX MODEL ==========
# ###########################################################################
print("\n\n" + "="*30 + " SARIMAX MODEL SECTION " + "="*30)

# ========== DATA CLEANING & PREPROCESSING (SARIMAX) ==========
print("\nCleaning and preprocessing data for SARIMAX...")
df_sarimax = df_original.copy()
df_sarimax = df_sarimax.dropna(subset=['Purchase Amount (USD)'])
df_sarimax['Review Rating'] = df_sarimax['Review Rating'].fillna(df_sarimax['Review Rating'].median()) # Median for SARIMAX
df_sarimax['Date Purchase'] = pd.to_datetime(df_sarimax['Date Purchase'])
df_sarimax = df_sarimax.sort_values('Date Purchase')

# Aggregate daily & add features for SARIMAX
df_daily_sarimax = df_sarimax.groupby('Date Purchase').agg({
    'Purchase Amount (USD)': 'mean', # Mean for SARIMAX
    'Item Purchased': 'count',
    'Review Rating': 'mean',
    'Payment Method': lambda x: (x == 'Credit Card').mean() # Example: ratio of credit card payments
}).rename(columns={'Payment Method': 'credit_card_ratio'})

# Add more sophisticated time features for SARIMAX
df_daily_sarimax['day_of_week'] = df_daily_sarimax.index.dayofweek
df_daily_sarimax['month'] = df_daily_sarimax.index.month
df_daily_sarimax['day_of_month'] = df_daily_sarimax.index.day
df_daily_sarimax['week_of_year'] = df_daily_sarimax.index.isocalendar().week.astype(int)
df_daily_sarimax['quarter'] = df_daily_sarimax.index.quarter
df_daily_sarimax['is_weekend'] = df_daily_sarimax.index.dayofweek.isin([5,6]).astype(int)
df_daily_sarimax['is_month_end'] = df_daily_sarimax.index.is_month_end.astype(int)

# More advanced moving averages for SARIMAX
df_daily_sarimax['MA3'] = df_daily_sarimax['Purchase Amount (USD)'].rolling(window=3).mean()
df_daily_sarimax['MA7'] = df_daily_sarimax['Purchase Amount (USD)'].rolling(window=7).mean()
df_daily_sarimax['MA14'] = df_daily_sarimax['Purchase Amount (USD)'].rolling(window=14).mean()
df_daily_sarimax['volatility'] = df_daily_sarimax['Purchase Amount (USD)'].rolling(window=7).std()

# Add lag features for SARIMAX
df_daily_sarimax['lag1'] = df_daily_sarimax['Purchase Amount (USD)'].shift(1)
df_daily_sarimax['lag7'] = df_daily_sarimax['Purchase Amount (USD)'].shift(7)
df_daily_sarimax['lag30'] = df_daily_sarimax['Purchase Amount (USD)'].shift(30)
df_daily_sarimax['ewm7'] = df_daily_sarimax['Purchase Amount (USD)'].ewm(span=7).mean()
df_daily_sarimax['ewm30'] = df_daily_sarimax['Purchase Amount (USD)'].ewm(span=30).mean()

# Fill missing values (bfill then ffill)
df_daily_sarimax = df_daily_sarimax.fillna(method='bfill').fillna(method='ffill')

# Transform target variable with log1p for SARIMAX
df_daily_sarimax['Purchase Amount (USD)_log1p'] = np.log1p(df_daily_sarimax['Purchase Amount (USD)'])

# Scale target variable separately for SARIMAX (on log-transformed data)
target_scaler_sarimax = StandardScaler()
df_daily_sarimax['Purchase Amount (USD)_scaled'] = target_scaler_sarimax.fit_transform(df_daily_sarimax[['Purchase Amount (USD)_log1p']])

# Enhanced exogenous variables for SARIMAX
exog_columns_sarimax = [
    'Item Purchased', 'Review Rating', 'credit_card_ratio',
    'day_of_week', 'day_of_month', 'week_of_year',
    'month', 'quarter', 'is_weekend', 'is_month_end',
    'MA3', 'MA7', 'MA14', 'volatility',
    'lag1', 'lag7', 'lag30', 'ewm7', 'ewm30'
]
# Ensure all exog columns exist and handle potential new NaNs from MAs/lags on copies
for col in exog_columns_sarimax:
    if col not in df_daily_sarimax.columns:
        print(f"Warning: Exogenous column {col} not found, filling with 0 or recheck logic.")
        df_daily_sarimax[col] = 0 # Placeholder, review if this occurs

df_daily_sarimax = df_daily_sarimax.dropna() # Drop rows with NaNs created by shifts/rolling windows

# Scale exogenous features for SARIMAX
exog_scaler_sarimax = StandardScaler()
df_daily_sarimax[exog_columns_sarimax] = exog_scaler_sarimax.fit_transform(df_daily_sarimax[exog_columns_sarimax])

# Visualize preprocessed data for SARIMAX
plt.figure(figsize=(15,5))
df_daily_sarimax['Purchase Amount (USD)_scaled'].hist(bins=50, color='purple', alpha=0.7)
plt.title('Preprocessed Daily Average Purchase Amount (SARIMAX - Log-Transformed & Scaled)')
plt.xlabel('Log-Transformed & Scaled Purchase Amount (USD)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== TRAIN-TEST SPLIT (SARIMAX - 80/20) ==========
print("\nSplitting SARIMAX data 80/20...")
n_sarimax = len(df_daily_sarimax)
train_size_sarimax = int(n_sarimax * 0.8)
train_data_sarimax = df_daily_sarimax.iloc[:train_size_sarimax]
test_data_sarimax = df_daily_sarimax.iloc[train_size_sarimax:]

# Visualize train-test split for SARIMAX
plt.figure(figsize=(15,7))
plt.plot(train_data_sarimax.index, train_data_sarimax['Purchase Amount (USD)_scaled'],
         label='Training Data (SARIMAX)', color='blue', alpha=0.7)
plt.plot(test_data_sarimax.index, test_data_sarimax['Purchase Amount (USD)_scaled'],
         label='Test Data (SARIMAX)', color='red', alpha=0.7)
plt.title('Train-Test Split Visualization (SARIMAX - Scaled Target)')
plt.xlabel('Date')
plt.ylabel('Log-Transformed & Scaled Purchase Amount (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

train_target_sarimax = train_data_sarimax['Purchase Amount (USD)_scaled']
train_exog_sarimax = train_data_sarimax[exog_columns_sarimax]
test_target_sarimax = test_data_sarimax['Purchase Amount (USD)_scaled'] # Keep for consistency if needed later, though actuals are from original scale
test_exog_sarimax = test_data_sarimax[exog_columns_sarimax]


# ========== SARIMAX MODEL ==========
print("\nTraining optimized SARIMAX model...")
param_combinations_sarimax = [
    ((3,1,3), (1,1,1,7)),
    ((2,1,2), (2,1,2,7)),
    ((3,1,2), (1,1,1,14)),
    ((4,1,1), (0,1,1,7)),
    ((2,0,2), (1,1,1,7)),
    ((1,1,1), (0,1,1,7))
]

best_aic_sarimax = float('inf')
best_model_sarimax_fitted = None
best_params_sarimax = None

for order, seasonal_order in param_combinations_sarimax:
    try:
        model_sarimax = SARIMAX(
            train_target_sarimax, # Use scaled target for training
            exog=train_exog_sarimax,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result_sarimax = model_sarimax.fit(disp=False, maxiter=200)
        if result_sarimax.aic < best_aic_sarimax:
            best_aic_sarimax = result_sarimax.aic
            best_model_sarimax_fitted = result_sarimax
            best_params_sarimax = (order, seasonal_order)
            print(f"New best SARIMAX params: {best_params_sarimax} with AIC: {best_aic_sarimax}")
    except Exception as e:
        # print(f"Error with params {order}, {seasonal_order}: {e}")
        continue

if best_model_sarimax_fitted is None:
    print("SARIMAX model fitting failed for all parameter combinations. Exiting SARIMAX section.")
    # Optionally, exit or handle this case, e.g. by fitting a default model
    # For now, we'll let it error out if no model is found, or you can add a default.
    # Example: Defaulting if no model found
    print("No suitable SARIMAX model found with given parameters. Attempting a default.")
    try:
        default_order, default_seasonal_order = ((1,1,1), (0,0,0,0)) # A very simple default
        model_sarimax = SARIMAX(train_target_sarimax, exog=train_exog_sarimax, order=default_order, seasonal_order=default_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        best_model_sarimax_fitted = model_sarimax.fit(disp=False, maxiter=100)
        best_params_sarimax = (default_order, default_seasonal_order)
        print(f"Using default SARIMAX params: {best_params_sarimax} with AIC: {best_model_sarimax_fitted.aic}")
    except Exception as e:
        print(f"Default SARIMAX model also failed: {e}")
        # If default also fails, then SARIMAX part cannot continue for predictions
        raise SystemExit("SARIMAX model could not be fitted.")


print(f"Best SARIMAX parameters: {best_params_sarimax}")

# Prepare future exogenous features for SARIMAX
future_dates_sarimax = pd.date_range(
    start=test_data_sarimax.index[-1] + pd.Timedelta(days=1),
    periods=7,
    freq='D'
)
future_exog_sarimax_df = pd.DataFrame(index=future_dates_sarimax, columns=exog_columns_sarimax)

# Populate time-based features for future dates
future_exog_sarimax_df['day_of_week'] = future_dates_sarimax.dayofweek
future_exog_sarimax_df['month'] = future_dates_sarimax.month
future_exog_sarimax_df['day_of_month'] = future_dates_sarimax.day
future_exog_sarimax_df['week_of_year'] = future_dates_sarimax.isocalendar().week.astype(int)
future_exog_sarimax_df['quarter'] = future_dates_sarimax.quarter
future_exog_sarimax_df['is_weekend'] = future_dates_sarimax.dayofweek.isin([5,6]).astype(int)
future_exog_sarimax_df['is_month_end'] = future_dates_sarimax.is_month_end.astype(int)

# Use last known values from df_daily_sarimax for other features (before scaling)
# This requires careful handling as some features depend on the target variable which is unknown for the future.
# For simplicity, we'll use the last available scaled values from df_daily_sarimax for these exog features.
# A more robust approach would be to predict these exog features or use their recent trends/averages.
last_known_exog_sarimax = df_daily_sarimax[exog_columns_sarimax].iloc[-1]

for col in exog_columns_sarimax:
    if col not in ['day_of_week', 'month', 'day_of_month', 'week_of_year', 'quarter', 'is_weekend', 'is_month_end']:
        # For other columns (like MAs, lags, etc.), we'd ideally forecast them or use a strategy.
        # Here, we'll use the last known scaled value from the original exog_scaler_sarimax transform.
        # This is a simplification. For MAs/lags, they should be iteratively updated if possible.
        future_exog_sarimax_df[col] = last_known_exog_sarimax[col] # Using last known *scaled* value

future_exog_sarimax_df = future_exog_sarimax_df.fillna(method='ffill').fillna(method='bfill') # Fill any remaining NaNs

# The scaler `exog_scaler_sarimax` was fit on df_daily_sarimax[exog_columns_sarimax].
# The future_exog_sarimax_df already contains scaled values for non-time based features
# and unscaled for time-based features. We need to scale only the time-based ones
# OR reconstruct future_exog_sarimax_df with unscaled values and then scale the whole dataframe.
# Let's reconstruct with unscaled values for clarity and then scale.

future_exog_unscaled_sarimax_df = pd.DataFrame(index=future_dates_sarimax, columns=exog_columns_sarimax)
future_exog_unscaled_sarimax_df['day_of_week'] = future_dates_sarimax.dayofweek
future_exog_unscaled_sarimax_df['month'] = future_dates_sarimax.month
future_exog_unscaled_sarimax_df['day_of_month'] = future_dates_sarimax.day
future_exog_unscaled_sarimax_df['week_of_year'] = future_dates_sarimax.isocalendar().week.astype(int)
future_exog_unscaled_sarimax_df['quarter'] = future_dates_sarimax.quarter
future_exog_unscaled_sarimax_df['is_weekend'] = future_dates_sarimax.dayofweek.isin([5,6]).astype(int)
future_exog_unscaled_sarimax_df['is_month_end'] = future_dates_sarimax.is_month_end.astype(int)

# For other features, use the last *unscaled* value from the original df_sarimax or df_daily_sarimax before scaling
last_unscaled_item_purchased = df_sarimax.set_index('Date Purchase')['Item Purchased'].resample('D').count().iloc[-1]
last_unscaled_review_rating = df_sarimax.set_index('Date Purchase')['Review Rating'].resample('D').mean().iloc[-1]
last_unscaled_credit_card_ratio = df_daily_sarimax['credit_card_ratio'].iloc[-1] # This was already a ratio

# For MAs, lags, volatility, ewm - these are more complex for future. Using last known value is a simplification.
# These should ideally be calculated based on predicted values iteratively if high accuracy is needed.
# For now, propagate last known original values before they were scaled.

# Example for 'Item Purchased', 'Review Rating', 'credit_card_ratio' (these are simpler exog)
future_exog_unscaled_sarimax_df['Item Purchased'] = df_daily_sarimax['Item Purchased'].iloc[-1] # Using last pre-scaled value
future_exog_unscaled_sarimax_df['Review Rating'] = df_daily_sarimax['Review Rating'].iloc[-1] # Using last pre-scaled value
future_exog_unscaled_sarimax_df['credit_card_ratio'] = df_daily_sarimax['credit_card_ratio'].iloc[-1] # Using last pre-scaled value

# For complex exog like MAs, lags, volatility: use last value from df_daily_sarimax *before* their own scaling
# This is a strong assumption.
for col in ['MA3', 'MA7', 'MA14', 'volatility', 'lag1', 'lag7', 'lag30', 'ewm7', 'ewm30']:
    if col in df_daily_sarimax.columns: # Check if it exists
         future_exog_unscaled_sarimax_df[col] = df_daily_sarimax[col].iloc[-1] # This is the last *scaled* value if scaled in place
                                                                            # Should be from df_daily_sarimax *before* exog scaling

# Correct approach:
# Store original (unscaled) exog values before scaling them, then use those last values.
# Or, for MAs/lags based on target, iteratively predict target and then exog.
# Given the script structure, we'll use the last values available in `df_daily_sarimax` for these columns *before* they were scaled by `exog_scaler_sarimax`.
# This means we need to be careful about the state of `df_daily_sarimax`.
# The current `df_daily_sarimax[exog_columns_sarimax]` are SCALED.
# We need their unscaled counterparts.
# Let's assume for simplicity, we take the last values from the *already scaled* `test_exog_sarimax` and hope the scaler handles it.
# This is not ideal but simpler for this merge.
# A truly robust way is to re-calculate them based on the *actual definitions* using the last `n` days of actuals and iteratively for future.

for col in exog_columns_sarimax:
    if future_exog_unscaled_sarimax_df[col].isnull().any():
        future_exog_unscaled_sarimax_df[col] = test_exog_sarimax[col].iloc[-1] # Fallback to last scaled exog value from test set

future_exog_unscaled_sarimax_df = future_exog_unscaled_sarimax_df.fillna(0) # Fill any remaining NaNs just in case

future_exog_sarimax_scaled = pd.DataFrame(
    exog_scaler_sarimax.transform(future_exog_unscaled_sarimax_df), # Scale the constructed future exog
    index=future_dates_sarimax,
    columns=exog_columns_sarimax
)

# Generate predictions for test period
pred_sarimax_scaled = best_model_sarimax_fitted.get_forecast(steps=len(test_data_sarimax), exog=test_exog_sarimax)
pred_mean_sarimax_scaled = pred_sarimax_scaled.predicted_mean

# Inverse transform predictions for test period
sarimax_pred_log1p = target_scaler_sarimax.inverse_transform(pred_mean_sarimax_scaled.values.reshape(-1,1))
sarimax_pred_values = np.expm1(sarimax_pred_log1p) # Inverse of log1p
sarimax_pred_values = np.maximum(0, sarimax_pred_values) # Ensure non-negative

# Prepare actual test values (inverse transform from original data)
actual_test_target_scaled = test_data_sarimax['Purchase Amount (USD)_scaled'].values.reshape(-1,1)
actual_test_log1p = target_scaler_sarimax.inverse_transform(actual_test_target_scaled)
actual_test_sarimax_values = np.expm1(actual_test_log1p)
actual_test_sarimax_values = np.maximum(0, actual_test_sarimax_values)

# Generate future predictions
future_pred_sarimax_scaled = best_model_sarimax_fitted.get_forecast(steps=7, exog=future_exog_sarimax_scaled) # Use scaled future exog
future_pred_mean_sarimax_scaled = future_pred_sarimax_scaled.predicted_mean

# Inverse transform future predictions
future_values_log1p = target_scaler_sarimax.inverse_transform(future_pred_mean_sarimax_scaled.values.reshape(-1,1))
future_values_sarimax = np.expm1(future_values_log1p)
future_values_sarimax = np.maximum(0, future_values_sarimax)

# Calculate metrics for SARIMAX
mae_metric_sarimax = mean_absolute_error(actual_test_sarimax_values, sarimax_pred_values)
rmse_metric_sarimax = np.sqrt(mean_squared_error(actual_test_sarimax_values, sarimax_pred_values))
r2_metric_sarimax = r2_score(actual_test_sarimax_values, sarimax_pred_values)
mape_metric_sarimax = mean_absolute_percentage_error(actual_test_sarimax_values, sarimax_pred_values) * 100

# ========== VISUALIZATION (SARIMAX) ==========
plt.figure(figsize=(15,7))
plt.plot(test_data_sarimax.index, actual_test_sarimax_values, label='Actual (SARIMAX)', color='black')
plt.plot(test_data_sarimax.index, sarimax_pred_values, label='Predicted (SARIMAX)', color='blue', alpha=0.8)
plt.plot(future_dates_sarimax, future_values_sarimax, label='Future Forecast (SARIMAX)', color='red', linestyle='--')
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
metrics_sarimax_names = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
metric_values_sarimax = [rmse_metric_sarimax, mae_metric_sarimax, mape_metric_sarimax, r2_metric_sarimax*100]
x_sarimax = np.arange(len(metrics_sarimax_names))
width_sarimax = 0.5
bars_sarimax = plt.bar(x_sarimax, metric_values_sarimax, width_sarimax, label='SARIMAX', color='green', alpha=0.7) # Changed color

for i, bar in enumerate(bars_sarimax):
    if i == 3:  # R² Score
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(metric_values_sarimax),
                 f'{metric_values_sarimax[i]/100:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(metric_values_sarimax),
                 f'{metric_values_sarimax[i]:.1f}', ha='center', va='bottom', fontweight='bold')

plt.title('SARIMAX Model Performance Metrics', fontsize=16, fontweight='bold')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks(x_sarimax, metrics_sarimax_names)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, max(metric_values_sarimax) * 1.1)
plt.figtext(0.5, 0.01,
            'Note: For RMSE, MAE, MAPE → lower is better | For R² → higher is better',
            ha='center', fontsize=10, style='italic')
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

# ========== RINGKASAN PERFORMA SARIMAX ==========
print(f"\n{'='*50}")
print("RINGKASAN PERFORMA SARIMAX:")
print(f"{'='*50}")
print(f"Best parameters: {best_params_sarimax}")
print(f"RMSE: {int(rmse_metric_sarimax)}")
print(f"MAE: {int(mae_metric_sarimax)}")
print(f"MAPE: {int(mape_metric_sarimax)}%")
print(f"R²: {r2_metric_sarimax:.2f}")
print(f"{'='*50}")

# Display 7-day forecast (SARIMAX)
print("\n7-DAY FORECAST (SARIMAX):")
print("="*50)
for date, pred_val in zip(future_dates_sarimax, future_values_sarimax):
    print(f"{date.strftime('%Y-%m-%d')}: ${pred_val[0]:.2f}")
print("="*50)

# Display last 37 days prediction details (SARIMAX)
print("\nLAST 37 DAYS PREDICTION DETAILS (SARIMAX):")
print("           Date      Actual    Predicted   Difference")
print("="*60)
last_37_actual_sarimax = actual_test_sarimax_values[-37:]
last_37_pred_sarimax = sarimax_pred_values[-37:]
last_37_dates_sarimax = test_data_sarimax.index[-37:]

for date, act, pred_val in zip(last_37_dates_sarimax, last_37_actual_sarimax, last_37_pred_sarimax):
    diff = pred_val[0] - act[0]
    print(f"{date.strftime('%Y-%m-%d')}  {act[0]:10.2f}  {pred_val[0]:10.2f}  {diff:10.2f}")
print("="*60)

# Optional: Save the SARIMAX model and results
print("\nSaving SARIMAX model and results...")
with open('sarimax_model_results.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model_sarimax_fitted,
        'params': best_params_sarimax,
        'predictions_test': sarimax_pred_values,
        'actual_test': actual_test_sarimax_values,
        'future_predictions': future_values_sarimax,
        'metrics': {
            'rmse': rmse_metric_sarimax,
            'mae': mae_metric_sarimax,
            'mape': mape_metric_sarimax,
            'r2': r2_metric_sarimax
        },
        'target_scaler': target_scaler_sarimax,
        'exog_scaler': exog_scaler_sarimax,
        'exog_columns': exog_columns_sarimax
    }, f)
print("SARIMAX model and results saved to 'sarimax_model_results.pkl'")

print("\n\n" + "="*30 + " SCRIPT EXECUTION COMPLETE " + "="*30)