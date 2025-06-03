import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read the dataset
df = pd.read_csv('Fashion_Retail_Sales.csv')

# Convert Date Purchase to datetime
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'])

# Create a figure with subplots
plt.figure(figsize=(20, 12))

# 1. Purchase Amount Distribution
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='Purchase Amount (USD)', bins=30)
plt.title('Distribution of Purchase Amounts')
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Count')

# 2. Review Rating Distribution
plt.subplot(2, 2, 2)
sns.histplot(data=df.dropna(subset=['Review Rating']), x='Review Rating', bins=20)
plt.title('Distribution of Review Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')

# 3. Payment Method Distribution
plt.subplot(2, 2, 3)
payment_counts = df['Payment Method'].value_counts()
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%')
plt.title('Payment Method Distribution')

# 4. Purchase Trend Over Time
plt.subplot(2, 2, 4)
df.set_index('Date Purchase')['Purchase Amount (USD)'].resample('M').mean().plot()
plt.title('Average Purchase Amount Trend (Monthly)')
plt.xlabel('Date')
plt.ylabel('Average Amount (USD)')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Additional Analysis: Correlation between Review Rating and Purchase Amount
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.dropna(subset=['Review Rating']), 
                x='Review Rating', 
                y='Purchase Amount (USD)')
plt.title('Correlation between Review Rating and Purchase Amount')
plt.show()

# Analisis Tren dan Musiman
plt.figure(figsize=(20, 12))

# 1. Tren Harian
plt.subplot(2, 2, 1)
daily_sales = df.groupby('Date Purchase')['Purchase Amount (USD)'].mean()
daily_sales.plot()
plt.title('Tren Penjualan Harian')
plt.xlabel('Tanggal')
plt.ylabel('Rata-rata Penjualan (USD)')

# 2. Tren Bulanan
plt.subplot(2, 2, 2)
monthly_sales = df.groupby(df['Date Purchase'].dt.to_period('M'))['Purchase Amount (USD)'].mean()
monthly_sales.plot(kind='bar')
plt.title('Tren Penjualan Bulanan')
plt.xlabel('Bulan')
plt.ylabel('Rata-rata Penjualan (USD)')
plt.xticks(rotation=45)

# 3. Pola Musiman berdasarkan Hari dalam Seminggu
plt.subplot(2, 2, 3)
df['Day of Week'] = df['Date Purchase'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_pattern = df.groupby('Day of Week')['Purchase Amount (USD)'].mean()
weekly_pattern = weekly_pattern.reindex(day_order)
weekly_pattern.plot(kind='bar')
plt.title('Pola Penjualan Berdasarkan Hari')
plt.xlabel('Hari')
plt.ylabel('Rata-rata Penjualan (USD)')
plt.xticks(rotation=45)

# 4. Pola Musiman berdasarkan Bulan
plt.subplot(2, 2, 4)
df['Month'] = df['Date Purchase'].dt.month_name()
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_pattern = df.groupby('Month')['Purchase Amount (USD)'].mean()
monthly_pattern = monthly_pattern.reindex(month_order)
monthly_pattern.plot(kind='bar')
plt.title('Pola Penjualan Berdasarkan Bulan')
plt.xlabel('Bulan')
plt.ylabel('Rata-rata Penjualan (USD)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Heatmap pola penjualan mingguan-bulanan
plt.figure(figsize=(12, 8))
df['DayOfWeek'] = df['Date Purchase'].dt.day_name()
df['Month'] = df['Date Purchase'].dt.month_name()
heatmap_data = df.pivot_table(
    values='Purchase Amount (USD)', 
    index='DayOfWeek',
    columns='Month',
    aggfunc='mean'
)
heatmap_data = heatmap_data.reindex(index=day_order)
heatmap_data = heatmap_data.reindex(columns=month_order)

sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('Heatmap Pola Penjualan (Hari vs Bulan)')
plt.ylabel('Hari')
plt.xlabel('Bulan')
plt.tight_layout()
plt.show()

# Analisis untuk Time Series Modeling
plt.figure(figsize=(20, 12))

# 1. Time Series Decomposition
daily_avg = df.groupby('Date Purchase')['Purchase Amount (USD)'].mean().reset_index()
daily_avg.set_index('Date Purchase', inplace=True)
decomposition = seasonal_decompose(daily_avg, period=30)

plt.subplot(4, 1, 1)
decomposition.observed.plot()
plt.title('Observed Data')

plt.subplot(4, 1, 2)
decomposition.trend.plot()
plt.title('Trend')

plt.subplot(4, 1, 3)
decomposition.seasonal.plot()
plt.title('Seasonal')

plt.subplot(4, 1, 4)
decomposition.resid.plot()
plt.title('Residual')

plt.tight_layout()
plt.show()

# 2. Data Split Visualization (70/20/10)
total_days = (df['Date Purchase'].max() - df['Date Purchase'].min()).days
train_end = df['Date Purchase'].min() + pd.Timedelta(days=int(total_days * 0.7))
valid_end = train_end + pd.Timedelta(days=int(total_days * 0.2))

plt.figure(figsize=(15, 6))
plt.plot(daily_avg.index[daily_avg.index <= train_end], 
         daily_avg[daily_avg.index <= train_end], 
         label='Training (70%)')
plt.plot(daily_avg.index[(daily_avg.index > train_end) & (daily_avg.index <= valid_end)], 
         daily_avg[(daily_avg.index > train_end) & (daily_avg.index <= valid_end)], 
         label='Validation (20%)')
plt.plot(daily_avg.index[daily_avg.index > valid_end], 
         daily_avg[daily_avg.index > valid_end], 
         label='Test (10%)')
plt.title('Data Split Visualization')
plt.xlabel('Date')
plt.ylabel('Average Purchase Amount (USD)')
plt.legend()
plt.show()

# 3. Moving Averages
plt.figure(figsize=(15, 6))

# Urutkan data berdasarkan tanggal
daily_avg = daily_avg.sort_index()

# Hitung MA dengan cara yang lebih tepat
ma7 = daily_avg['Purchase Amount (USD)'].rolling(window=7, center=True).mean()
ma30 = daily_avg['Purchase Amount (USD)'].rolling(window=30, center=True).mean()

# Plot dengan warna dan style yang lebih jelas
plt.plot(daily_avg.index, daily_avg['Purchase Amount (USD)'], 
         label='Daily Average', alpha=0.3, color='gray')
plt.plot(daily_avg.index, ma7, 
         label='7-day MA', linewidth=2, color='blue')
plt.plot(daily_avg.index, ma30, 
         label='30-day MA', linewidth=2, color='red')

plt.title('Moving Averages Analysis')
plt.xlabel('Date')
plt.ylabel('Average Purchase Amount (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4. Autocorrelation Plot
plt.figure(figsize=(15, 6))
plot_acf(daily_avg.dropna(), lags=30)
plt.title('Autocorrelation Analysis')
plt.show()

# 5. Distribution of Daily Changes
daily_changes = daily_avg.diff().dropna()
plt.figure(figsize=(10, 6))
sns.histplot(daily_changes, bins=30)
plt.title('Distribution of Daily Changes in Purchase Amount')
plt.xlabel('Change in Average Purchase Amount (USD)')
plt.ylabel('Frequency')
plt.show()
