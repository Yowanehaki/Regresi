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

# Rolling window diperpanjang untuk smoothing lebih halus
df_daily['Purchase Amount (USD)'] = df_daily['Purchase Amount (USD)'].rolling(window=7, center=True).mean().fillna(df_daily['Purchase Amount (USD)'])

# ========== NORMALISASI DATA ==========
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(df_daily[['Purchase Amount (USD)']])

# ========== PERSIAPAN DATA UNTUK GRU ==========
n_steps = 60  # Diperpanjang untuk capture pola jangka panjang
X, y = [], []
for i in range(n_steps, len(scaled_values)):
    X.append(scaled_values[i-n_steps:i])
    y.append(scaled_values[i])
X, y = np.array(X), np.array(y)

# ========== TRAIN-TEST SPLIT ==========
train_size = int(len(df_daily) * 0.8)
X_train, X_test = X[:train_size - n_steps], X[train_size - n_steps:]
y_train, y_test = y[:train_size - n_steps], y[train_size - n_steps:]
train_sarimax = df_daily['Purchase Amount (USD)'].iloc[:train_size]
test_sarimax = df_daily['Purchase Amount (USD)'].iloc[train_size:]

# ========== OPTIMASI MODEL SARIMAX YANG DIPERLUAS ==========
print("Training Enhanced SARIMAX Model...")

# Transformasi log untuk stabilisasi varian
train_sarimax_log = np.log1p(train_sarimax)
test_sarimax_log = np.log1p(test_sarimax)
best_aic = float('inf')
best_sarimax_model = None
best_params = None

# Parameter kombinasi yang lebih komprehensif dengan seasonal pattern
param_combinations = [
    ((0,1,1), (0,1,1,7)),    # Simple ARIMA dengan seasonal
    ((0,1,2), (0,1,1,7)),    # MA model
    ((1,1,0), (1,1,0,7)),    # AR model
    ((1,1,1), (1,1,1,7)),    # Basic ARIMA
    ((2,1,1), (1,1,1,7)),    # Extended AR
    ((1,1,2), (1,1,1,7)),    # Extended MA
    ((2,1,2), (1,1,1,7)),    # Extended ARIMA
    ((1,1,1), (2,1,1,7)),    # Extended seasonal AR
    ((1,1,1), (1,1,2,7)),    # Extended seasonal MA
    ((2,1,2), (2,1,2,7)),    # Complex model
    ((3,1,2), (1,1,1,7)),    # Higher order AR
    ((2,1,3), (1,1,1,7)),    # Higher order MA
    ((1,1,1), (0,1,1,14)),   # Bi-weekly seasonality
    ((2,1,2), (1,1,1,14)),   # Bi-weekly with complex ARIMA
    ((1,1,1), (1,1,1,30)),   # Monthly seasonality
]

print("Searching for best SARIMAX parameters...")
for i, (order, seasonal_order) in enumerate(param_combinations):
    try:
        temp_model = SARIMAX(
            train_sarimax_log, 
            order=order, 
            seasonal_order=seasonal_order,
            enforce_stationarity=False, 
            enforce_invertibility=False,
            trend='c'  # Include constant
        )
        temp_result = temp_model.fit(
            maxiter=1000, 
            method='lbfgs', 
            disp=False,
            low_memory=True
        )
        if temp_result.aic < best_aic:
            best_aic = temp_result.aic
            best_sarimax_model = temp_result
            best_params = (order, seasonal_order)
            print(f"New best AIC: {best_aic:.2f} with params {order}, {seasonal_order}")
    except Exception as e:
        continue

# Fallback model jika tidak ada yang berhasil
if best_sarimax_model is None:
    print("Using fallback SARIMAX model...")
    sarimax_model = SARIMAX(
        train_sarimax_log, 
        order=(1,1,1), 
        seasonal_order=(1,1,1,7),
        trend='c'
    )
    sarimax_result = sarimax_model.fit(maxiter=1000, disp=False)
    best_params = ((1,1,1), (1,1,1,7))
else:
    sarimax_result = best_sarimax_model

print(f"Best SARIMAX parameters: {best_params}")
print(f"Final AIC: {sarimax_result.aic:.2f}")

# Prediksi dengan confidence interval
forecast_result = sarimax_result.get_forecast(steps=len(test_sarimax))
sarimax_pred_log = forecast_result.predicted_mean
sarimax_pred_ci = forecast_result.conf_int()

# Inverse transform dari log
sarimax_pred = np.expm1(sarimax_pred_log)
sarimax_pred_lower = np.expm1(sarimax_pred_ci.iloc[:, 0])
sarimax_pred_upper = np.expm1(sarimax_pred_ci.iloc[:, 1])

# Clipping untuk menghindari nilai negatif
sarimax_pred = np.maximum(sarimax_pred, 0)

# ========== MEMBANGUN MODEL GRU ==========
print("Training Enhanced GRU Model...")

model = Sequential([
    GRU(128, return_sequences=True, input_shape=(n_steps, 1),
        dropout=0.3, recurrent_dropout=0.2),
    BatchNormalization(),
    GRU(64, return_sequences=True,
        dropout=0.3, recurrent_dropout=0.2),
    BatchNormalization(),
    GRU(32,
        dropout=0.3, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(1)
])

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])

# ========== CALLBACK FUNCTIONS ==========
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=7, min_lr=1e-7)

# ========== TRAINING MODEL GRU ==========
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ========== PREDIKSI DAN INVERSE TRANSFORM ==========
y_pred_gru = model.predict(X_test)
y_pred_gru_inv = scaler.inverse_transform(y_pred_gru)
y_test_inv = scaler.inverse_transform(y_test)

# ========== EVALUASI MODEL SARIMAX ==========
mae_sarimax = mean_absolute_error(test_sarimax, sarimax_pred)
mape_sarimax = mean_absolute_percentage_error(test_sarimax, sarimax_pred) * 100
r2_sarimax = r2_score(test_sarimax, sarimax_pred)
mse_sarimax = mean_squared_error(test_sarimax, sarimax_pred)
rmse_sarimax = np.sqrt(mse_sarimax)

# ========== EVALUASI MODEL GRU ==========
mae_gru = mean_absolute_error(y_test_inv, y_pred_gru_inv)
mape_gru = mean_absolute_percentage_error(y_test_inv, y_pred_gru_inv) * 100
r2_gru = r2_score(y_test_inv, y_pred_gru_inv)
mse_gru = mean_squared_error(y_test_inv, y_pred_gru_inv)
rmse_gru = np.sqrt(mse_gru)

# ========== MENAMPILKAN HASIL EVALUASI ==========
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

# ========== VISUALISASI 1: PERBANDINGAN PREDIKSI DENGAN CONFIDENCE INTERVAL ==========
plt.figure(figsize=(16,8))
plt.plot(test_sarimax.index, test_sarimax, label='Actual', linewidth=2.5, color='black')
plt.plot(test_sarimax.index, sarimax_pred, label='SARIMAX', linewidth=2, color='blue', alpha=0.8)
plt.fill_between(test_sarimax.index, sarimax_pred_lower, sarimax_pred_upper, 
                 color='blue', alpha=0.2, label='SARIMAX 95% CI')
plt.plot(test_sarimax.index[-len(y_pred_gru):], y_pred_gru_inv, label='GRU', linewidth=2, color='green', alpha=0.8)
plt.title("Perbandingan Prediksi SARIMAX vs GRU (Optimized)", fontsize=16, fontweight='bold')
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Pembelian (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== VISUALISASI 2: BAR CHART PERFORMA ==========
plt.figure(figsize=(10, 6))
metrics = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
sarimax_values = [rmse_sarimax, mae_sarimax, mape_sarimax, r2_sarimax*100]
gru_values = [rmse_gru, mae_gru, mape_gru, r2_gru*100]
x = np.arange(len(metrics))
width = 0.35

bars1 = plt.bar(x - width/2, sarimax_values, width, label='SARIMAX (Optimized)', color='blue', alpha=0.7)
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

plt.title('Perbandingan Performa Model SARIMAX vs GRU (Optimized)', fontsize=16, fontweight='bold')
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