import numpy as np 
import pandas as pd 
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import r2_score
import os

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(60, return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(LSTM(80, return_sequences=True))
    model.add(Dropout(0.4))
    
    model.add(LSTM(120))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    return model

def main():
    # Data parameters
    start = '2010-01-01'
    end = '2024-01-15'
    symbol = 'AAPL'
    
    print(f"Downloading data for {symbol}...")
    df = yf.download(symbol, start=start, end=end)
    
    # Print column names for debugging
    print("Available columns:", df.columns.tolist())
    
    # Extract closing prices
    closing_prices = df['Close'].values.reshape(-1, 1)
    
    # Split data
    train_size = int(len(closing_prices) * 0.70)
    data_training = closing_prices[:train_size]
    data_testing = closing_prices[train_size:]
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_scaled = scaler.fit_transform(data_training)
    
    # Prepare sequences
    x_train = []
    y_train = []
    
    for i in range(100, len(data_training_scaled)):
        x_train.append(data_training_scaled[i-100:i])
        y_train.append(data_training_scaled[i, 0])
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    print("Creating and training model...")
    model = create_model((100, 1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Save model
    model_path = os.path.join('Code', 'keras_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Prepare test data
    past_100_days = data_training[-100:]
    final_df = np.concatenate([past_100_days, data_testing])
    input_data = scaler.transform(final_df)
    
    x_test = []
    y_test = []
    
    for i in range(100, len(input_data)):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # Make predictions
    y_predicted = model.predict(x_test)
    
    # Scale back predictions
    scale_factor = 1/scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor
    
    # Calculate R-squared score
    r_squared = r2_score(y_test, y_predicted)
    print(f"R-squared score: {r_squared:.4f}")

if __name__ == "__main__":
    main()





