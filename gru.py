# ========== IMPORT LIBRARY ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
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

# Visualize time series data
plt.figure(figsize=(15,7))
plt.plot(df_daily.index, df_daily['Purchase Amount (USD)'], 
         label='Preprocessed Data', color='blue', alpha=0.7)
plt.title('Preprocessed Time Series Data')
plt.xlabel('Date')
plt.ylabel('Scaled Purchase Amount') 
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== TRAIN-TEST SPLIT (80/20) ==========
print("\nSplitting data 80/20...")
n = len(df_daily)
train_size = int(n * 0.8)
train_data = df_daily.iloc[:train_size]
test_data = df_daily.iloc[train_size:]

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

# Split 80/20
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]  # Changed from X_val, y_val

print(f"GRU training data: {X_train.shape}")
print(f"GRU testing data: {X_test.shape}")  # Updated print message

# ========== MEMBANGUN MODEL GRU ==========
print("\nBuilding GRU model...")
model = Sequential([
    # Increase complexity and add regularization
    GRU(128, return_sequences=True, input_shape=(n_steps, X.shape[2]), 
        recurrent_dropout=0.2),
    GRU(64, return_sequences=True, recurrent_dropout=0.2),
    GRU(32, return_sequences=False, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  # Change to linear for regression
])

# Use a lower learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])  # Huber loss is more robust
model.summary()

# ========== CALLBACK FUNCTIONS ==========
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,  # Increase patience
    restore_best_weights=True,
    min_delta=0.00001,  # More sensitive
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,  # More aggressive LR reduction
    patience=15,
    min_lr=1e-7,
    verbose=1
)

# ========== TRAINING MODEL GRU ==========
print("\nTraining GRU model...")
history = model.fit(
    X_train, y_train,
    epochs=300,  # Increase epochs
    batch_size=32,  # Larger batch size
    validation_data=(X_test, y_test),  # Changed from X_val, y_val
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ========== PREDIKSI DAN INVERSE TRANSFORM GRU ==========
print("\nGenerating GRU predictions...")
y_pred_gru = model.predict(X_test)  # Changed from X_val
y_pred_gru_inv = scaler.inverse_transform(y_pred_gru)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))  # Changed from y_val

# ========== EVALUASI GRU ==========
mae_gru = mean_absolute_error(y_test_inv, y_pred_gru_inv)  # Changed from y_val_inv
mape_gru = mean_absolute_percentage_error(y_test_inv, y_pred_gru_inv) * 100
r2_gru = r2_score(y_test_inv, y_pred_gru_inv)
rmse_gru = np.sqrt(mean_squared_error(y_test_inv, y_pred_gru_inv))

print("\n=== GRU Evaluation ===")
print(f"RMSE  : {rmse_gru:.2f}")
print(f"MAE   : {mae_gru:.2f}")
print(f"MAPE  : {mape_gru:.2f}%")
print(f"R2    : {r2_gru:.4f}")

# ========== PREDIKSI 7 HARI KEDEPAN ==========
print("\nPredicting next 7 days...")
last_sequence = X_test[-1:]  # Changed from X_val
future_predictions = []

for _ in range(7):
    # Get prediction for next day
    next_pred = model.predict(last_sequence)
    future_predictions.append(next_pred[0, 0])
    
    # Update the sequence for next prediction
    new_seq = last_sequence[0][1:].copy()
    new_row = last_sequence[0][-1].copy()
    new_row[0] = next_pred[0, 0]  # Update only the target value
    last_sequence = np.append(new_seq, [new_row], axis=0).reshape(1, n_steps, X.shape[2])

# Inverse transform predictions
future_pred_inv = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create future dates
future_dates = pd.date_range(start=test_data.index[-1], periods=8)[1:]  # Changed from val_data

print("\nPredicted values for next 7 days:")
for date, pred in zip(future_dates, future_pred_inv):
    print(f"{date.date()}: ${pred[0]:.2f}")

# ========== DETAIL VISUALISASI =========
# Show sample data
print("\nSample of actual data:")
print(df.head())

# Show last 37 days comparison (80/20 split visualization)
plt.figure(figsize=(15,7))
last_37_actual = y_test_inv[-37:]  # Changed from y_val_inv
last_37_pred = y_pred_gru_inv[-37:]
last_37_dates = test_data.index[-37:]  # Changed from val_data

plt.plot(last_37_dates, last_37_actual, 
         label='Actual', linewidth=2, color='black')
plt.plot(last_37_dates, last_37_pred, 
         label='Predicted', linewidth=2, color='blue', alpha=0.8)
plt.plot(future_dates, future_pred_inv, 
         label='Future Predictions', linewidth=2, color='red', linestyle='--')

plt.title('Last 37 Days of Predictions + 7 Days Future Forecast')
plt.xlabel('Date')
plt.ylabel('Purchase Amount (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Show prediction details for last 37 days
print("\nLast 37 days prediction details:")
comparison_df = pd.DataFrame({
    'Date': last_37_dates,
    'Actual': last_37_actual.flatten(),
    'Predicted': last_37_pred.flatten(),
    'Difference': abs(last_37_actual.flatten() - last_37_pred.flatten())
})
print(comparison_df)

# Save model
model.save('gru_model.h5')
print("\nGRU model saved to 'gru_model.h5'")

# ========== VISUALIZATION =========
plt.figure(figsize=(15,7))
# Adjust index to match predictions length
aligned_test_dates = test_data.index[-len(y_test_inv):]

plt.plot(aligned_test_dates, np.maximum(0, y_test_inv), label='Actual', color='black')
plt.plot(aligned_test_dates, np.maximum(0, y_pred_gru_inv), label='Predicted', color='blue', alpha=0.8)
plt.plot(future_dates, future_pred_inv, label='Future Forecast', color='red', linestyle='--')
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
metrics = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
gru_values = [rmse_gru, mae_gru, mape_gru, r2_gru*100]
x = np.arange(len(metrics))
width = 0.5
bars = plt.bar(x, gru_values, width, label='GRU', color='blue', alpha=0.7)

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