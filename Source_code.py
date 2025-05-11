import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the stock ticker symbol and date range
ticker = "AAPL"  # Example: Apple Inc.
start_date = "2020-01-01"
end_date = "2023-01-01"

# Download historical stock data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Select the 'Close' price as the target variable
df = data[["Close"]]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
df["Close"] = scaler.fit_transform(np.array(df["Close"]).reshape(-1, 1))

# Create training and testing datasets
training_size = int(len(df) * 0.8)  # 80% for training
train_data, test_data = df[0:training_size], df[training_size:len(df)]

# Create the dataset with a lookback period
def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# Reshape for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=1)

# Make predictions and evaluate
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Inverse scaling

# Evaluate (e.g., using RMSE)
# ...
