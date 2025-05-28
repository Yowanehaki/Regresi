# ========== IMPORT LIBRARY ==========
# Library untuk manipulasi data dan visualisasi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Library untuk preprocessing dan evaluasi model
from sklearn.preprocessing import MinMaxScaler  # Untuk normalisasi data
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error

# Library untuk model SARIMAX (time series tradisional)
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Library untuk deep learning model GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Mengabaikan warning untuk output yang lebih bersih
import warnings
warnings.filterwarnings('ignore')

# ========== LOAD DAN EKSPLORASI DATA ==========
# Memuat dataset penjualan fashion retail
df = pd.read_csv('Fashion_Retail_Sales.csv')
df.head()  # Melihat 5 baris pertama data

df.info()  # Informasi struktur data (tipe data, missing values)

df.describe(include="all")  # Statistik deskriptif untuk semua kolom

# ========== DATA CLEANING ==========
# Menghapus baris yang memiliki nilai kosong pada kolom Purchase Amount
df = df.dropna(subset=['Purchase Amount (USD)'])

# Mengisi nilai kosong pada Review Rating dengan nilai rata-rata
df['Review Rating'] = df['Review Rating'].fillna(df['Review Rating'].mean())

# Mengkonversi kolom tanggal ke format datetime dan mengurutkan data
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'])
df = df.sort_values('Date Purchase')

# ========== PREPROCESSING DATA ==========
# Mengubah data menjadi time series harian dengan agregasi:
# - Purchase Amount: dijumlahkan per hari
# - Item Purchased: dihitung jumlahnya per hari  
# - Review Rating: dirata-ratakan per hari
df_daily = df.set_index('Date Purchase').resample('D').agg({
    'Purchase Amount (USD)': 'sum',
    'Item Purchased': 'count',
    'Review Rating': 'mean'
}).fillna(0)

# Menghaluskan data dengan rolling average untuk mengurangi noise
# Window=3 artinya rata-rata bergerak 3 hari
df_daily['Purchase Amount (USD)'] = df_daily['Purchase Amount (USD)'].rolling(window=3, center=True).mean().fillna(df_daily['Purchase Amount (USD)'])

# ========== NORMALISASI DATA ==========
# Normalisasi data ke rentang 0-1 untuk training neural network
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(df_daily[['Purchase Amount (USD)']])

# ========== PERSIAPAN DATA UNTUK GRU ==========
# Membuat sequences untuk supervised learning
# n_steps = jumlah hari sebelumnya yang digunakan untuk prediksi
n_steps = 45  # Menggunakan 45 hari sebelumnya untuk prediksi hari berikutnya
X, y = [], []  # X = input sequences, y = target values

# Loop untuk membuat sequences
for i in range(n_steps, len(scaled_values)):
    X.append(scaled_values[i-n_steps:i])  # 45 hari sebelumnya
    y.append(scaled_values[i])  # hari yang akan diprediksi
X, y = np.array(X), np.array(y)

# ========== TRAIN-TEST SPLIT ==========
# Membagi data menjadi 80% training dan 20% testing
train_size = int(len(df_daily) * 0.8)

# Split untuk GRU model
X_train, X_test = X[:train_size - n_steps], X[train_size - n_steps:]
y_train, y_test = y[:train_size - n_steps], y[train_size - n_steps:]

# Split untuk SARIMAX model (menggunakan data asli, bukan yang dinormalisasi)
train_sarimax = df_daily['Purchase Amount (USD)'].iloc[:train_size]
test_sarimax = df_daily['Purchase Amount (USD)'].iloc[train_size:]

# ========== OPTIMASI MODEL SARIMAX ==========
print("Training Enhanced SARIMAX Model...")

# Grid search untuk mencari parameter terbaik SARIMAX
# Parameter order: (p,d,q) - autoregressive, differencing, moving average
# Parameter seasonal_order: (P,D,Q,S) - seasonal components dengan periode S=7 hari
best_aic = float('inf')  # AIC = Akaike Information Criterion (semakin kecil semakin baik)
best_sarimax_model = None

# Kombinasi parameter yang akan dicoba
param_combinations = [
    ((1,1,1), (1,1,1,7)),  # Parameter dasar
    ((2,1,2), (1,1,1,7)),  # Lebih kompleks di non-seasonal
    ((1,1,2), (2,1,1,7)),  # Lebih kompleks di seasonal
    ((2,1,1), (1,1,2,7)),  # Kombinasi lain
]

# Mencoba setiap kombinasi parameter
for order, seasonal_order in param_combinations:
    try:
        temp_model = SARIMAX(train_sarimax, order=order, seasonal_order=seasonal_order)
        temp_result = temp_model.fit(maxiter=500, method='lbfgs', disp=False)
        # Simpan model dengan AIC terkecil
        if temp_result.aic < best_aic:
            best_aic = temp_result.aic
            best_sarimax_model = temp_result
    except:
        continue  # Skip jika ada error dalam fitting

# Jika tidak ada model yang berhasil, gunakan parameter default
if best_sarimax_model is None:
    sarimax_model = SARIMAX(train_sarimax, order=(1,1,1), seasonal_order=(1,1,1,7))
    sarimax_result = sarimax_model.fit(maxiter=500, method='lbfgs', disp=False)
else:
    sarimax_result = best_sarimax_model

# Membuat prediksi untuk periode test
sarimax_pred = sarimax_result.get_forecast(steps=len(test_sarimax)).predicted_mean

# ========== MEMBANGUN MODEL GRU ==========
print("Training Enhanced GRU Model...")

# Arsitektur neural network dengan 3 layer GRU
model = Sequential([
    # Layer GRU pertama: 128 units, return sequences untuk layer berikutnya
    GRU(128, return_sequences=True, input_shape=(n_steps, 1), 
        dropout=0.2, recurrent_dropout=0.2),  # Dropout untuk mencegah overfitting
    
    # Batch normalization untuk stabilitas training
    BatchNormalization(),
    
    # Layer GRU kedua: 64 units
    GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    BatchNormalization(),
    
    # Layer GRU ketiga: 32 units, tidak return sequences
    GRU(32, dropout=0.2, recurrent_dropout=0.2),
    
    # Dense layer untuk pemrosesan akhir
    Dense(16, activation='relu'),
    Dropout(0.3),  # Dropout rate lebih tinggi
    
    # Output layer
    Dense(1)  # 1 output untuk prediksi nilai tunggal
])

# Optimizer dengan learning rate yang disesuaikan
optimizer = Adam(learning_rate=0.001)

# Compile model dengan loss function Huber (robust terhadap outliers)
model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])

# ========== CALLBACK FUNCTIONS ==========
# Early stopping: berhenti training jika tidak ada improvement
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning rate scheduler: kurangi learning rate jika stuck
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

# ========== TRAINING MODEL GRU ==========
# Training dengan 30 epochs dan validation split
history = model.fit(
    X_train, y_train, 
    epochs=30,  # Jumlah epoch training
    batch_size=32,  # Ukuran batch untuk gradient update
    validation_split=0.2,  # 20% dari training data untuk validasi
    callbacks=[early_stopping, reduce_lr],  # Callback functions
    verbose=1  # Tampilkan progress training
)

# ========== PREDIKSI DAN INVERSE TRANSFORM ==========
# Melakukan prediksi pada data test
y_pred_gru = model.predict(X_test)

# Mengembalikan hasil prediksi ke skala asli (denormalisasi)
y_pred_gru_inv = scaler.inverse_transform(y_pred_gru)
y_test_inv = scaler.inverse_transform(y_test)

# ========== EVALUASI MODEL SARIMAX ==========
# Menghitung berbagai metrik evaluasi untuk SARIMAX
mae_sarimax = mean_absolute_error(test_sarimax, sarimax_pred)  # Mean Absolute Error
mape_sarimax = mean_absolute_percentage_error(test_sarimax, sarimax_pred) * 100  # Mean Absolute Percentage Error
r2_sarimax = r2_score(test_sarimax, sarimax_pred)  # R-squared Score
mse_sarimax = mean_squared_error(test_sarimax, sarimax_pred)  # Mean Squared Error
rmse_sarimax = np.sqrt(mse_sarimax)  # Root Mean Squared Error

# ========== EVALUASI MODEL GRU ==========
# Menghitung berbagai metrik evaluasi untuk GRU
mae_gru = mean_absolute_error(y_test_inv, y_pred_gru_inv)
mape_gru = mean_absolute_percentage_error(y_test_inv, y_pred_gru_inv) * 100
r2_gru = r2_score(y_test_inv, y_pred_gru_inv)
mse_gru = mean_squared_error(y_test_inv, y_pred_gru_inv)
rmse_gru = np.sqrt(mse_gru)

# ========== MENAMPILKAN HASIL EVALUASI ==========
print("=== Enhanced SARIMAX Evaluation ===")
print(f"RMSE  : {rmse_sarimax:.2f}")  # Semakin kecil semakin baik
print(f"MAE   : {mae_sarimax:.2f}")   # Semakin kecil semakin baik
print(f"MAPE  : {mape_sarimax:.2f}%") # Semakin kecil semakin baik
print(f"R2    : {r2_sarimax:.4f}")    # Semakin besar semakin baik (max=1)

print("\n=== Enhanced GRU Evaluation ===")
print(f"RMSE  : {rmse_gru:.2f}")
print(f"MAE   : {mae_gru:.2f}")
print(f"MAPE  : {mape_gru:.2f}%")
print(f"R2    : {r2_gru:.4f}")

# ========== VISUALISASI 1: PERBANDINGAN PREDIKSI ==========
# Plot untuk membandingkan prediksi kedua model dengan data aktual
plt.figure(figsize=(14,6))

# Plot data aktual
plt.plot(test_sarimax.index, test_sarimax, label='Actual', 
         linewidth=2.5, color='black')

# Plot prediksi SARIMAX
plt.plot(test_sarimax.index, sarimax_pred, label='SARIMAX', 
         linewidth=2, color='blue', alpha=0.8)

# Plot prediksi GRU (hanya untuk periode yang sama dengan SARIMAX)
plt.plot(test_sarimax.index[-len(y_pred_gru):], y_pred_gru_inv, 
         label='GRU', linewidth=2, color='green', alpha=0.8)

plt.title("Perbandingan Prediksi SARIMAX vs GRU", fontsize=16, fontweight='bold')
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Pembelian (USD)")
plt.legend()
plt.grid(True, alpha=0.3)  # Grid dengan transparansi
plt.xticks(rotation=45)  # Rotasi label tanggal
plt.tight_layout()
plt.show()

# ========== VISUALISASI 2: BAR CHART PERFORMA ==========
# Membuat bar chart untuk perbandingan metrik evaluasi
plt.figure(figsize=(10, 6))

# Data untuk bar chart
metrics = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
sarimax_values = [rmse_sarimax, mae_sarimax, mape_sarimax, r2_sarimax*100]  # R2 dikali 100 untuk skala
gru_values = [rmse_gru, mae_gru, mape_gru, r2_gru*100]

# Posisi bar
x = np.arange(len(metrics))
width = 0.35

# Membuat bar chart
bars1 = plt.bar(x - width/2, sarimax_values, width, label='SARIMAX', color='blue', alpha=0.7)
bars2 = plt.bar(x + width/2, gru_values, width, label='GRU', color='green', alpha=0.7)

# Menambahkan label nilai pada setiap bar
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    if i == 3:  # R² score (perlu dibagi 100 untuk tampilan)
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

# Catatan interpretasi metrik
plt.figtext(0.5, 0.02, 
           'Catatan: Untuk RMSE, MAE, MAPE → semakin kecil semakin baik | Untuk R² → semakin besar semakin baik', 
           ha='center', fontsize=10, style='italic')
plt.tight_layout()
plt.show()

# ========== RINGKASAN PERFORMA ==========
# Menampilkan ringkasan model mana yang lebih baik
print(f"\n{'='*50}")
print("RINGKASAN PERFORMA:")
print(f"{'='*50}")
better_rmse = "SARIMAX" if rmse_sarimax < rmse_gru else "GRU"
better_r2 = "SARIMAX" if r2_sarimax > r2_gru else "GRU"
print(f"Model dengan RMSE terbaik: {better_rmse}")
print(f"Model dengan R² terbaik: {better_r2}")
print(f"{'='*50}")