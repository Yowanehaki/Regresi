# ========== IMPORT LIBRARY ==========
#100 EPOCHS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ========== LOAD DAN EKSPLORASI DATA ==========
df = pd.read_csv('Fashion_Retail_Sales.csv')
df.head()
df.info()
df.describe(include="all")

# ========== DATA CLEANING ==========
df = df.dropna(subset=['Purchase Amount (USD)'])
df['Review Rating'] = df['Review Rating'].fillna(df['Review Rating'].mean())
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'])
df = df.sort_values('Date Purchase')

# Convert date to datetime features
df['Year'] = df['Date Purchase'].dt.year
df['Month'] = df['Date Purchase'].dt.month
df['Day'] = df['Date Purchase'].dt.day
df['DayOfWeek'] = df['Date Purchase'].dt.dayofweek

# Create category mappings
payment_mapping = {'Cash': 0, 'Credit Card': 1}
df['Payment_Coded'] = df['Payment Method'].map(payment_mapping)

# Encode item categories
item_mapping = {item: idx for idx, item in enumerate(df['Item Purchased'].unique())}
df['Item_Coded'] = df['Item Purchased'].map(item_mapping)

# Prepare features
features = ['Item_Coded', 'Review Rating', 'Payment_Coded', 
           'Year', 'Month', 'Day', 'DayOfWeek']
target = 'Purchase Amount (USD)'

# Split data into features (X) and target (y)
X = df[features]
y = df[target]

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Prepare time series data properly
daily_sales = df.groupby('Date Purchase')['Purchase Amount (USD)'].mean().reset_index()
daily_sales = daily_sales.set_index('Date Purchase')
daily_sales = daily_sales.sort_index()

# Split the data into training and testing sets
train_size = int(len(daily_sales) * 0.8)
train_data = daily_sales[:train_size]
test_data = daily_sales[train_size:]

# ========== FUNGSI UNTUK GRID SEARCH ==========
def find_best_arima_params(train_series, max_p=3, max_d=2, max_q=3):
    """Find best ARIMA parameters using AIC"""
    best_aic = float('inf')
    best_params = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(train_series, order=(p, d, q))
                    result = model.fit()
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_params = (p, d, q)
                except:
                    continue
    return best_params, best_aic

def find_best_sarima_params(train_series, max_p=2, max_d=1, max_q=2, s=7):
    """Find best SARIMA parameters using AIC"""
    best_aic = float('inf')
    best_params = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(2):
                    for D in range(2):
                        for Q in range(2):
                            try:
                                model = SARIMAX(train_series, 
                                              order=(p, d, q),
                                              seasonal_order=(P, D, Q, s),
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
                                result = model.fit(disp=False)
                                if result.aic < best_aic:
                                    best_aic = result.aic
                                    best_params = ((p, d, q), (P, D, Q, s))
                            except:
                                continue
    return best_params, best_aic

# ========== 1. ARIMA MODEL ==========
print("Training ARIMA Model...")
best_arima_params, best_arima_aic = find_best_arima_params(train_data['Purchase Amount (USD)'])
if best_arima_params is None:
    best_arima_params = (1, 1, 1)  # fallback
    
arima_model = ARIMA(train_data['Purchase Amount (USD)'], order=best_arima_params)
arima_fitted = arima_model.fit()
arima_pred = arima_fitted.forecast(steps=len(test_data))
print(f"Best ARIMA params: {best_arima_params}, AIC: {arima_fitted.aic:.2f}")

# ========== 2. ARIMAX MODEL ==========
print("Training ARIMAX Model...")
# ARIMAX adalah ARIMA dengan exogenous variables
try:
    arimax_model = ARIMA(train_data['Purchase Amount (USD)'], 
                        exog=train_exog,
                        order=best_arima_params)
    arimax_fitted = arimax_model.fit()
    arimax_pred = arimax_fitted.forecast(steps=len(test_data), exog=test_exog)
    print(f"ARIMAX params: {best_arima_params}, AIC: {arimax_fitted.aic:.2f}")
except:
    print("ARIMAX failed, using ARIMA prediction")
    arimax_pred = arima_pred
    arimax_fitted = arima_fitted

# ========== 3. SARIMA MODEL ==========
print("Training SARIMA Model...")
best_sarima_params, best_sarima_aic = find_best_sarima_params(train_data['Purchase Amount (USD)'])
if best_sarima_params is None:
    best_sarima_params = ((1, 1, 1), (1, 1, 1, 7))  # fallback

sarima_model = SARIMAX(train_data['Purchase Amount (USD)'],
                      order=best_sarima_params[0],
                      seasonal_order=best_sarima_params[1],
                      enforce_stationarity=False,
                      enforce_invertibility=False)
sarima_fitted = sarima_model.fit(disp=False)
sarima_forecast = sarima_fitted.get_forecast(steps=len(test_data))
sarima_pred = sarima_forecast.predicted_mean
print(f"Best SARIMA params: {best_sarima_params}, AIC: {sarima_fitted.aic:.2f}")

# ========== 4. SARIMAX MODEL (ORIGINAL) ==========
print("Training Enhanced SARIMAX Model with Exogenous Variables...")
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
    except Exception as e:
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

print(f"Best SARIMAX params: {best_params}, AIC: {best_sarimax_model.aic:.2f}")

# SARIMAX PREDICTION
forecast = best_sarimax_model.get_forecast(steps=len(test_data), exog=test_exog)
sarimax_pred = forecast.predicted_mean
sarimax_pred_lower = forecast.conf_int().iloc[:, 0]
sarimax_pred_upper = forecast.conf_int().iloc[:, 1]

# ========== 5. LSTM MODEL ==========
print("Training LSTM Model...")

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(n_steps, X.shape[2])),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(32),
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dense(1)
])

lstm_optimizer = Adam(learning_rate=0.001)
lstm_model.compile(optimizer=lstm_optimizer, loss='mse', metrics=['mae'])

# LSTM Training
lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, min_delta=0.0001),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
    ],
    verbose=1
)

# LSTM Prediction
y_pred_lstm = lstm_model.predict(X_test)

# ========== 6. GRU MODEL (ORIGINAL) ==========
print("Training Enhanced GRU Model...")

gru_model = Sequential([
    GRU(64, return_sequences=True, input_shape=(n_steps, X.shape[2])),
    BatchNormalization(),
    Dropout(0.2),
    GRU(32),
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dense(1)
])

gru_optimizer = Adam(learning_rate=0.001)
gru_model.compile(optimizer=gru_optimizer, loss='mse', metrics=['mae'])

# GRU Training
gru_history = gru_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, min_delta=0.0001),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
    ],
    verbose=1
)

# GRU Prediction
y_pred_gru = gru_model.predict(X_test)

# ========== INVERSE TRANSFORM PREDICTIONS ==========
# Statistical models
arima_pred_inv = scaler.inverse_transform(arima_pred.values.reshape(-1,1)).flatten()
arimax_pred_inv = scaler.inverse_transform(arimax_pred.values.reshape(-1,1)).flatten()
sarima_pred_inv = scaler.inverse_transform(sarima_pred.values.reshape(-1,1)).flatten()
sarimax_pred_inv = scaler.inverse_transform(sarimax_pred.values.reshape(-1,1)).flatten()
sarimax_pred_lower_inv = scaler.inverse_transform(sarimax_pred_lower.values.reshape(-1,1)).flatten()
sarimax_pred_upper_inv = scaler.inverse_transform(sarimax_pred_upper.values.reshape(-1,1)).flatten()

# Neural network models
y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm)
y_pred_gru_inv = scaler.inverse_transform(y_pred_gru)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
actual_test = scaler.inverse_transform(test_data['Purchase Amount (USD)'].values.reshape(-1,1)).flatten()

# ========== EVALUASI SEMUA MODEL ==========
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    r2 = r2_score(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

# Calculate metrics for all models
metrics_arima = calculate_metrics(actual_test, arima_pred_inv)
metrics_arimax = calculate_metrics(actual_test, arimax_pred_inv)
metrics_sarima = calculate_metrics(actual_test, sarima_pred_inv)
metrics_sarimax = calculate_metrics(actual_test, sarimax_pred_inv)
metrics_lstm = calculate_metrics(y_test_inv.flatten(), y_pred_lstm_inv.flatten())
metrics_gru = calculate_metrics(y_test_inv.flatten(), y_pred_gru_inv.flatten())

# Print all metrics
models = ['ARIMA', 'ARIMAX', 'SARIMA', 'SARIMAX', 'LSTM', 'GRU']
all_metrics = [metrics_arima, metrics_arimax, metrics_sarima, metrics_sarimax, metrics_lstm, metrics_gru]

print("\n" + "="*80)
print("EVALUASI SEMUA MODEL")
print("="*80)
for model, metrics in zip(models, all_metrics):
    print(f"\n=== {model} Evaluation ===")
    print(f"RMSE  : {metrics['RMSE']:.2f}")
    print(f"MAE   : {metrics['MAE']:.2f}")
    print(f"MAPE  : {metrics['MAPE']:.2f}%")
    print(f"R2    : {metrics['R2']:.4f}")

# ========== VISUALISASI KOMPREHENSIF ==========

# Plot 1: Individual Model Predictions (2x3 subplot)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

predictions = [arima_pred_inv, arimax_pred_inv, sarima_pred_inv, 
              sarimax_pred_inv, y_pred_lstm_inv.flatten(), y_pred_gru_inv.flatten()]
colors = ['red', 'orange', 'purple', 'blue', 'brown', 'green']

for i, (model, pred, color, metrics) in enumerate(zip(models, predictions, colors, all_metrics)):
    axes[i].plot(test_data.index, actual_test, label='Actual', linewidth=2, color='black')
    if model == 'LSTM' or model == 'GRU':
        axes[i].plot(test_data.index[-len(pred):], pred, 
                    label=f'{model} Prediction', linewidth=2, color=color, alpha=0.8)
    else:
        axes[i].plot(test_data.index, pred, 
                    label=f'{model} Prediction', linewidth=2, color=color, alpha=0.8)
    
    # Add confidence interval for SARIMAX
    if model == 'SARIMAX':
        axes[i].fill_between(test_data.index, sarimax_pred_lower_inv, sarimax_pred_upper_inv, 
                           color=color, alpha=0.1, label='95% CI')
    
    axes[i].set_title(f'{model} Model Prediction', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Purchase Amount (USD)')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    axes[i].tick_params(axis='x', rotation=45)
    
    # Add metrics text
    metrics_text = f'RMSE: {metrics["RMSE"]:.2f}\nR²: {metrics["R2"]:.4f}'
    axes[i].text(0.02, 0.95, metrics_text, transform=axes[i].transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.show()

# Plot 2: Combined Comparison
plt.figure(figsize=(18, 10))
plt.plot(test_data.index, actual_test, label='Actual', linewidth=3, color='black')

# Plot statistical models
plt.plot(test_data.index, arima_pred_inv, label='ARIMA', linewidth=2, color='red', alpha=0.7, linestyle='--')
plt.plot(test_data.index, arimax_pred_inv, label='ARIMAX', linewidth=2, color='orange', alpha=0.7, linestyle='--')
plt.plot(test_data.index, sarima_pred_inv, label='SARIMA', linewidth=2, color='purple', alpha=0.7, linestyle='--')
plt.plot(test_data.index, sarimax_pred_inv, label='SARIMAX', linewidth=2, color='blue', alpha=0.8)

# Plot neural network models
plt.plot(test_data.index[-len(y_pred_lstm_inv):], y_pred_lstm_inv, 
         label='LSTM', linewidth=2, color='brown', alpha=0.8)
plt.plot(test_data.index[-len(y_pred_gru_inv):], y_pred_gru_inv, 
         label='GRU', linewidth=2, color='green', alpha=0.8)

plt.title('Comparison of All Forecasting Models', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Purchase Amount (USD)', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 3: Performance Comparison Bar Chart
plt.figure(figsize=(15, 10))
metrics_names = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
x = np.arange(len(metrics_names))
width = 0.13

# Prepare data for plotting
rmse_values = [m['RMSE'] for m in all_metrics]
mae_values = [m['MAE'] for m in all_metrics]
mape_values = [m['MAPE'] for m in all_metrics]
r2_values = [m['R2']*100 for m in all_metrics]  # Convert to percentage for better visualization

all_values = [rmse_values, mae_values, mape_values, r2_values]

# Create bars
for i, (model, color) in enumerate(zip(models, colors)):
    values = [all_values[j][i] for j in range(4)]
    bars = plt.bar(x + i*width, values, width, label=model, color=color, alpha=0.7)
    
    # Add value labels on bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        if j == 3:  # R² score
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height/100:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)

plt.title('Performance Comparison of All Forecasting Models', fontsize=16, fontweight='bold')
plt.xlabel('Evaluation Metrics', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.xticks(x + width*2.5, metrics_names)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, axis='y')
plt.figtext(0.5, 0.02,
           'Note: For RMSE, MAE, MAPE → lower is better | For R² → higher is better',
           ha='center', fontsize=10, style='italic')
plt.tight_layout()
plt.show()

# Plot 4: Model Ranking
plt.figure(figsize=(12, 8))
# Calculate average rank for each model (lower rank = better performance)
ranks = {}
for model in models:
    ranks[model] = []

# Rank models for each metric (1 = best, 6 = worst)
for metric in ['RMSE', 'MAE', 'MAPE']:
    values = [all_metrics[i][metric] for i in range(len(models))]
    sorted_indices = np.argsort(values)  # ascending order (lower is better)
    for rank, idx in enumerate(sorted_indices):
        ranks[models[idx]].append(rank + 1)

# For R², higher is better
r2_values = [all_metrics[i]['R2'] for i in range(len(models))]
sorted_indices = np.argsort(r2_values)[::-1]  # descending order (higher is better)
for rank, idx in enumerate(sorted_indices):
    ranks[models[idx]].append(rank + 1)

# Calculate average rank
avg_ranks = {model: np.mean(ranks[model]) for model in models}
sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])

model_names = [item[0] for item in sorted_models]
rank_values = [item[1] for item in sorted_models]

bars = plt.bar(model_names, rank_values, color=['gold', 'silver', '#CD7F32', 'lightblue', 'lightcoral', 'lightgray'])
plt.title('Model Ranking (Based on Average Performance Rank)', fontsize=16, fontweight='bold')
plt.xlabel('Models', fontsize=12)
plt.ylabel('Average Rank (Lower is Better)', fontsize=12)
plt.ylim(0, 7)

# Add rank labels
for bar, rank in zip(bars, rank_values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{rank:.2f}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ========== RINGKASAN AKHIR ==========
print(f"\n{'='*80}")
print("RINGKASAN PERFORMA SEMUA MODEL:")
print(f"{'='*80}")

print("\nModel Ranking (berdasarkan rata-rata peringkat):")
for i, (model, rank) in enumerate(sorted_models, 1):
    print(f"{i}. {model}: {rank:.2f}")

print(f"\nModel terbaik secara keseluruhan: {sorted_models[0][0]}")

# Find best model for each metric
best_rmse = min(all_metrics, key=lambda x: x['RMSE'])
best_mae = min(all_metrics, key=lambda x: x['MAE'])
best_mape = min(all_metrics, key=lambda x: x['MAPE'])
best_r2 = max(all_metrics, key=lambda x: x['R2'])

rmse_idx = all_metrics.index(best_rmse)
mae_idx = all_metrics.index(best_mae)
mape_idx = all_metrics.index(best_mape)
r2_idx = all_metrics.index(best_r2)

print(f"\nBest RMSE: {models[rmse_idx]} ({best_rmse['RMSE']:.2f})")
print(f"Best MAE: {models[mae_idx]} ({best_mae['MAE']:.2f})")
print(f"Best MAPE: {models[mape_idx]} ({best_mape['MAPE']:.2f}%)")
print(f"Best R²: {models[r2_idx]} ({best_r2['R2']:.4f})")
print(f"{'='*80}")

# --- Prepare time series data for SARIMAX with exogenous variables ---
# Assume df is already cleaned and 'Date Purchase' is datetime

# Create daily aggregates and exogenous features
daily = df.groupby('Date Purchase').agg({
    'Purchase Amount (USD)': 'mean',
    'Review Rating': 'mean',
    'Payment Method': lambda x: (x == 'Credit Card').mean()
}).reset_index()

daily['Date Purchase'] = pd.to_datetime(daily['Date Purchase'])
daily['Month'] = daily['Date Purchase'].dt.month
daily['Day'] = daily['Date Purchase'].dt.day
daily['DayOfWeek'] = daily['Date Purchase'].dt.dayofweek
daily.set_index('Date Purchase', inplace=True)
daily = daily.sort_index()

# Split into train and test
train_size = int(len(daily) * 0.8)
train = daily.iloc[:train_size]
test = daily.iloc[train_size:]

# Define exogenous variables
exog_cols = ['Review Rating', 'Payment Method', 'Month', 'Day', 'DayOfWeek']
train_exog = train[exog_cols].fillna(0)
test_exog = test[exog_cols].fillna(0)

# --- SARIMAX Modeling ---
from statsmodels.tsa.statespace.sarimax import SARIMAX

print("Training Enhanced SARIMAX Model with Exogenous Variables...")
try:
    sarimax_model = SARIMAX(
        train['Purchase Amount (USD)'],
        exog=train_exog,
        order=(0, 1, 2),
        seasonal_order=(0, 1, 1, 7)
    )
    sarimax_results = sarimax_model.fit(disp=False)
    predictions = sarimax_results.forecast(steps=len(test), exog=test_exog)

    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
    mae = mean_absolute_error(test['Purchase Amount (USD)'], predictions)
    mape = mean_absolute_percentage_error(test['Purchase Amount (USD)'], predictions)
    rmse = mean_squared_error(test['Purchase Amount (USD)'], predictions, squared=False)
    r2 = r2_score(test['Purchase Amount (USD)'], predictions)

    print("\nSARIMAX Model Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2%}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test['Purchase Amount (USD)'], label='Actual', color='blue')
    plt.plot(test.index, predictions, label='Predicted', color='red', linestyle='--')
    plt.title('SARIMAX: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Purchase Amount (USD)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error in SARIMAX modeling: {str(e)}")