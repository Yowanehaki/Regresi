# ========== IMPORT LIBRARY ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

# Definisi feature columns
exog_columns = ['Item Purchased', 'Review Rating', 'day_of_week', 'month', 'quarter', 'is_weekend', 'MA7', 'MA30']

# Scaling features
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

# ========== PERSIAPAN DATA UNTUK GRU ==========
print("\nPreparing data for GRU...")
# Untuk GRU: Gabungkan target dan exog sebagai input
gru_features = ['Purchase Amount (USD)'] + exog_columns
gru_data = df_daily[gru_features].values

# Sequence length untuk GRU
n_steps = 14
print(f"Using sequence length of {n_steps} for GRU")

# Create sequences
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

print(f"GRU training data: {X_train.shape}")
print(f"GRU validation data: {X_val.shape}")
print(f"GRU test data: {X_test.shape}")

# ========== MEMBANGUN MODEL GRU ==========
print("\nBuilding GRU model...")
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
model.summary()

# ========== CALLBACK FUNCTIONS ==========
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    min_delta=0.0001,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# ========== TRAINING MODEL GRU ==========
print("\nTraining GRU model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ========== VISUALISASI LEARNING CURVES ==========
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

# ========== PREDIKSI DAN INVERSE TRANSFORM GRU ==========
print("\nGenerating GRU predictions...")
y_pred_gru = model.predict(X_test)
y_pred_gru_inv = scaler.inverse_transform(y_pred_gru)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# For comparison, also get actual test data
actual_test = scaler.inverse_transform(test_data['Purchase Amount (USD)'].values.reshape(-1,1)).flatten()

# ========== EVALUASI GRU ==========
mae_gru = mean_absolute_error(y_test_inv, y_pred_gru_inv)
mape_gru = mean_absolute_percentage_error(y_test_inv, y_pred_gru_inv) * 100
r2_gru = r2_score(y_test_inv, y_pred_gru_inv)
rmse_gru = np.sqrt(mean_squared_error(y_test_inv, y_pred_gru_inv))

print("\n=== GRU Evaluation ===")
print(f"RMSE  : {rmse_gru:.2f}")
print(f"MAE   : {mae_gru:.2f}")
print(f"MAPE  : {mape_gru:.2f}%")
print(f"R2    : {r2_gru:.4f}")

# ========== VISUALISASI HASIL PREDIKSI GRU ==========
print("\nVisualizing GRU results...")
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

# ========== VISUALISASI METRIK PERFORMA GRU ==========
print("\nVisualizing GRU performance metrics...")
plt.figure(figsize=(10, 6))
metrics = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
gru_values = [rmse_gru, mae_gru, mape_gru, r2_gru*100]
x = np.arange(len(metrics))
width = 0.5
bars = plt.bar(x, gru_values, width, label='GRU', color='green', alpha=0.7)

for i, bar in enumerate(bars):
    if i == 3:  # R² Score
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{gru_values[i]/100:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{gru_values[i]:.1f}', ha='center', va='bottom', fontweight='bold')

plt.title('GRU Model Performance Metrics', fontsize=16, fontweight='bold')
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

# ========== RINGKASAN PERFORMA GRU ==========
print(f"\n{'='*50}")
print("RINGKASAN PERFORMA GRU:")
print(f"{'='*50}")
print(f"Sequence length: {n_steps}")
print(f"RMSE: {rmse_gru:.2f}")
print(f"MAE: {mae_gru:.2f}")
print(f"MAPE: {mape_gru:.2f}%")
print(f"R²: {r2_gru:.4f}")
print(f"{'='*50}")

# Optional: Save the model and results
print("\nSaving GRU model and results...")
model.save('gru_model.h5')
import pickle
with open('gru_model_results.pkl', 'wb') as f:
    pickle.dump({
        'predictions': y_pred_gru_inv,
        'actual': y_test_inv,
        'metrics': {
            'rmse': rmse_gru,
            'mae': mae_gru,
            'mape': mape_gru,
            'r2': r2_gru
        }
    }, f)
print("GRU model saved to 'gru_model.h5'")
print("GRU results saved to 'gru_model_results.pkl'")

# Optional: Load saved results to compare
try:
    import pickle
    with open('sarimax_model_results.pkl', 'rb') as f:
        sarimax_results = pickle.load(f)
    
    # Plot comparison if SARIMAX results are available
    print("\nGenerating comparison with SARIMAX results...")
    sarimax_pred = sarimax_results['predictions']
    
    # Make sure to align the dates for proper comparison
    # Since GRU predictions might have different length due to sequence requirement
    test_dates_gru = test_data.index[-len(y_pred_gru_inv):]
    
    plt.figure(figsize=(15,7))
    plt.plot(test_dates_gru, y_test_inv, label='Actual', linewidth=2, color='black')
    plt.plot(test_dates_gru, y_pred_gru_inv, label='GRU', linewidth=2, color='green', alpha=0.8)
    
    # Plot SARIMAX predictions but only for the overlapping dates
    if len(test_dates_gru) <= len(sarimax_pred):
        sarimax_pred_aligned = sarimax_pred[-len(test_dates_gru):]
        plt.plot(test_dates_gru, sarimax_pred_aligned, label='SARIMAX', linewidth=2, color='blue', alpha=0.8)
    
    plt.title('Comparison of GRU and SARIMAX Predictions', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Purchase Amount (USD)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Create bar chart comparison
    plt.figure(figsize=(10, 6))
    metrics = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
    sarimax_metrics = [
        sarimax_results['metrics']['rmse'],
        sarimax_results['metrics']['mae'],
        sarimax_results['metrics']['mape'],
        sarimax_results['metrics']['r2']*100
    ]
    gru_metrics = [rmse_gru, mae_gru, mape_gru, r2_gru*100]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = plt.bar(x - width/2, sarimax_metrics, width, label='SARIMAX', color='blue', alpha=0.7)
    bars2 = plt.bar(x + width/2, gru_metrics, width, label='GRU', color='green', alpha=0.7)
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if i == 3:  # R² Score
            plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
                    f'{sarimax_metrics[i]/100:.3f}', ha='center', va='bottom', fontweight='bold')
            plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                    f'{gru_metrics[i]/100:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
                    f'{sarimax_metrics[i]:.1f}', ha='center', va='bottom', fontweight='bold')
            plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                    f'{gru_metrics[i]:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Comparison of Model Performance', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Print comparison summary
    print(f"\n{'='*50}")
    print("MODEL COMPARISON SUMMARY:")
    print(f"{'='*50}")
    better_rmse = "SARIMAX" if sarimax_metrics[0] < rmse_gru else "GRU"
    better_r2 = "SARIMAX" if sarimax_metrics[3]/100 > r2_gru else "GRU"
    print(f"Model with better RMSE: {better_rmse}")
    print(f"Model with better R²: {better_r2}")
    print(f"{'='*50}")
except:
    print("SARIMAX results not found. Run the SARIMAX program first for model comparison.")