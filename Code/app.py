import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
from sklearn.metrics import r2_score
import datetime
import os

# Configure page
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")

def load_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found for the ticker symbol. Please check if it's valid.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    st.title('Stock Trend Prediction')
    
    # Date inputs
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.date(2010, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.date.today())

    # Stock ticker input
    user_input = st.text_input('Enter Stock Ticker (e.g., AAPL, GOOGL)', 'AAPL')
    
    if not os.path.exists('Code/keras_model.h5'):
        st.error("Model file not found. Please train the model first using main.py")
        return

    df = load_data(user_input, start_date, end_date)
    
    if df is None:
        return

    # Describing Data
    st.subheader('Data Statistics')
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    # Moving averages
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    
    st.subheader('Closing Price vs Time Chart with Moving Averages')
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100, 'r', label='MA100')
    plt.plot(ma200, 'g', label='MA200')
    plt.plot(df.Close, 'b', label='Original Price')
    plt.legend()
    st.pyplot(fig)

    # Prepare data for prediction
    train_size = int(len(df) * 0.70)
    data_training = pd.DataFrame(df['Close'][:train_size])
    data_testing = pd.DataFrame(df['Close'][train_size:])

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit_transform(data_training)

    # Load model
    try:
        model = load_model('Code/keras_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Prepare test data
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
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

    # Scale back to original
    scale_factor = 1/scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Calculate R-squared score
    r_squared = r2_score(y_test, y_predicted)
    st.subheader('Model Performance')
    st.write(f'R-squared Score: {r_squared:.4f}')

    # Plot predictions
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time (days)')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

if __name__ == "__main__":
    main()