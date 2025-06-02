# ========== IMPORT LIBRARY ==========
#100 EPOCHS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
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

# ========== PREPROCESSING DATA ==========
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

# Scaling exogenous features (penting untuk SARIMAX)
exog_scaler = MinMaxScaler()
df_daily[exog_columns] = exog_scaler.fit_transform(df_daily[exog_columns])

# Scaling target
scaler = MinMaxScaler()
df_daily['Purchase Amount (USD)'] = scaler.fit_transform(df_daily[['Purchase Amount (USD)']])

# ========== TRAIN-VAL-TEST SPLIT (70/20/10) ==========
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

# Untuk GRU: Gabungkan target dan exog sebagai input
gru_features = ['Purchase Amount (USD)'] + exog_columns
gru_data = df_daily[gru_features].values

n_steps = 14
X, y = [], []
for i in range(n_steps, len(gru_data)):
    X.append(gru_data[i-n_steps:i])
    y.append(gru_data[i, 0])  # target tetap kolom pertama
X, y = np.array(X), np.array(y)

# Sinkronisasi split GRU
split1 = int(len(X) * 0.7)
split2 = int(len(X) * 0.9)
X_train, y_train = X[:split1], y[:split1]
X_val, y_val = X[split1:split2], y[split1:split2]
X_test, y_test = X[split2:], y[split2:]

# ========== SARIMAX PARAMETER TUNING & TRAINING ==========
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
print("SARIMAX exogenous variables:", exog_columns)

# ========== SARIMAX PREDICTION ==========
forecast = best_sarimax_model.get_forecast(steps=len(test_data), exog=test_exog)
sarimax_pred = forecast.predicted_mean
sarimax_pred_lower = forecast.conf_int().iloc[:, 0]
sarimax_pred_upper = forecast.conf_int().iloc[:, 1]

# Inverse scaling
sarimax_pred = scaler.inverse_transform(sarimax_pred.values.reshape(-1,1)).flatten()
sarimax_pred_lower = scaler.inverse_transform(sarimax_pred_lower.values.reshape(-1,1)).flatten()
sarimax_pred_upper = scaler.inverse_transform(sarimax_pred_upper.values.reshape(-1,1)).flatten()
actual_test = scaler.inverse_transform(test_data['Purchase Amount (USD)'].values.reshape(-1,1)).flatten()

# ========== MEMBANGUN MODEL GRU ==========
print("Training Enhanced GRU Model...")

model = Sequential([
    GRU(64, return_sequences=True, input_shape=(n_steps, X.shape[2])),
    BatchNormalization(),
    Dropout(0.2),
    GRU(32),
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# ========== CALLBACK FUNCTIONS ==========
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    min_delta=0.0001
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=1e-6
)

# ========== TRAINING MODEL GRU ==========
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ========== PREDIKSI DAN INVERSE TRANSFORM GRU ==========
y_pred_gru = model.predict(X_test)
y_pred_gru_inv = scaler.inverse_transform(y_pred_gru)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# ========== EVALUASI ==========
mae_sarimax = mean_absolute_error(actual_test, sarimax_pred)
mape_sarimax = mean_absolute_percentage_error(actual_test, sarimax_pred) * 100
r2_sarimax = r2_score(actual_test, sarimax_pred)
rmse_sarimax = np.sqrt(mean_squared_error(actual_test, sarimax_pred))

mae_gru = mean_absolute_error(y_test_inv, y_pred_gru_inv)
mape_gru = mean_absolute_percentage_error(y_test_inv, y_pred_gru_inv) * 100
r2_gru = r2_score(y_test_inv, y_pred_gru_inv)
rmse_gru = np.sqrt(mean_squared_error(y_test_inv, y_pred_gru_inv))

print("=== Enhanced SARIMAX Evaluation ===")
print(f"RMSE  : {rmse_sarimax:.2f}")
print(f"MAE   : {mae_sarimax:.2f}")
print(f"MAPE  : {mape_sarimax:.2f}%")
print(f"R2    : {r2_sarimax:.4f}")
print("\n=== Enhanced GRU Evaluation ===")
print(f"RMSE  : {rmse_gru:.2f}")
print(f"MAE   : {mae_gru:.2f}")
print(f"MAPE  : {mape_gru:.2f}%")
print(f"R2    : {r2_gru:.4f}")

# ========== VISUALISASI HASIL PREDIKSI ==========
# Plot 1: SARIMAX Prediction
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

# Plot 2: GRU Prediction
plt.figure(figsize=(15,7))
plt.plot(test_data.index[-len(y_pred_gru_inv):], y_test_inv, 
         label='Actual', linewidth=2, color='black')
plt.plot(test_data.index[-len(y_pred_gru_inv):], y_pred_gru_inv, 
         label='GRU Prediction', linewidth=2, color='green', alpha=0.8)
plt.title('GRU Model Prediction Results', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Purchase Amount (USD)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.text(0.02, 0.95, f'RMSE: {rmse_gru:.2f}\nR²: {r2_gru:.4f}', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()

# Plot 3: Combined Predictions
plt.figure(figsize=(15,7))
plt.plot(test_data.index, actual_test, label='Actual', linewidth=2, color='black')
plt.plot(test_data.index, sarimax_pred, label='SARIMAX', linewidth=2, color='blue', alpha=0.8)
plt.plot(test_data.index[-len(y_pred_gru_inv):], y_pred_gru_inv, 
         label='GRU', linewidth=2, color='green', alpha=0.8)
plt.title('Comparison of SARIMAX and GRU Predictions', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Purchase Amount (USD)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
text = f'SARIMAX Metrics:\nRMSE: {rmse_sarimax:.2f}\nR²: {r2_sarimax:.4f}\n\n'
text += f'GRU Metrics:\nRMSE: {rmse_gru:.2f}\nR²: {r2_gru:.4f}'
plt.text(0.02, 0.85, text, transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()

# ========== VISUALISASI 2: BAR CHART PERFORMA ==========
plt.figure(figsize=(10, 6))
metrics = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
sarimax_values = [rmse_sarimax, mae_sarimax, mape_sarimax, r2_sarimax*100]
gru_values = [rmse_gru, mae_gru, mape_gru, r2_gru*100]
x = np.arange(len(metrics))
width = 0.35
bars1 = plt.bar(x - width/2, sarimax_values, width, label='SARIMAX', color='blue', alpha=0.7)
bars2 = plt.bar(x + width/2, gru_values, width, label='GRU', color='green', alpha=0.7)
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    if i == 3:
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
                 f'{sarimax_values[i]/100:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                 f'{gru_values[i]/100:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
                 f'{sarimax_values[i]:.1f}', ha='center', va='bottom', fontweight='bold')
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                 f'{gru_values[i]:.1f}', ha='center', va='bottom', fontweight='bold')
plt.title('Perbandingan Performa Model SARIMAX vs GRU', fontsize=16, fontweight='bold')
plt.xlabel('Metrik Evaluasi')
plt.ylabel('Nilai')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.figtext(0.5, 0.02,
           'Catatan: Untuk RMSE, MAE, MAPE → semakin kecil semakin baik | Untuk R² → semakin besar semakin baik',
           ha='center', fontsize=10, style='italic')
plt.tight_layout()
plt.show()

# ========== RINGKASAN PERFORMA ==========
print(f"\n{'='*50}")
print("RINGKASAN PERFORMA:")
print(f"{'='*50}")
better_rmse = "SARIMAX" if rmse_sarimax < rmse_gru else "GRU"
better_r2 = "SARIMAX" if r2_sarimax > r2_gru else "GRU"
print(f"Model dengan RMSE terbaik: {better_rmse}")
print(f"Model dengan R² terbaik: {better_r2}")
print(f"Improvement SARIMAX: Using best params {best_params}")
print(f"{'='*50}")