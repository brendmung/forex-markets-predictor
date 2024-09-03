import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import time
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration parameters
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
SEQ_LENGTH = 30
PREDICTION_INTERVAL = 60  # seconds

# Load the trained model and scaler
class ForexLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ForexLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).detach().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).detach().to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ForexLSTM(input_dim=14, hidden_dim=28, num_layers=2, output_dim=1).to(device)
model.load_state_dict(torch.load('forex_model.pth', map_location=device))
model.eval()

scaler = joblib.load('forex_scaler.joblib')

def add_technical_indicators(df):
    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
    
    df['TR'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift()), 
                                     abs(df['low'] - df['close'].shift())))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (tp - ma) / (0.015 * mad)
    
    return df

def fetch_live_data():
    if not mt5.initialize():
        logging.error("MT5 initialization failed")
        return None

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, SEQ_LENGTH + 1)
    if rates is None:
        logging.error("Failed to fetch data")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = add_technical_indicators(df)
    df = df.dropna()

    mt5.shutdown()
    return df

def prepare_data(df):
    data = df[['open', 'high', 'low', 'close', 'tick_volume', 
               'EMA10', 'EMA30', 'MACD', 'Signal_Line', 'RSI', 
               'BB_upper', 'BB_lower', 'ATR', 'CCI']].values
    scaled_data = scaler.transform(data)
    X = np.array([scaled_data[:-1]])
    return X, data[-1]

def make_prediction(X):
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        prediction = model(X_tensor).cpu().numpy()
    return prediction[0][0]

def calculate_accuracy(actual, predicted, threshold=0.0001):
    return 100 if abs(actual - predicted) <= threshold * actual else 0

def main():
    correct_predictions = 0
    total_predictions = 0
    correct_direction_predictions = 0
    total_direction_predictions = 0
    actual_prices = []
    predicted_prices = []
    
    # Calculate and store the initial offset
    df_initial = fetch_live_data()
    X_initial, last_data_initial = prepare_data(df_initial)
    initial_prediction = make_prediction(X_initial)
    dummy_array = np.zeros((1, X_initial.shape[2]))
    dummy_array[0, 3] = initial_prediction
    inverse_transformed = scaler.inverse_transform(last_data_initial.reshape(1, -1) + dummy_array)
    initial_predicted_price = inverse_transformed[0, 3]
    initial_actual_price = df_initial['close'].iloc[-1]
    offset = initial_predicted_price - initial_actual_price

    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()

    while True:
        df = fetch_live_data()
        if df is None:
            time.sleep(PREDICTION_INTERVAL)
            continue

        X, last_data = prepare_data(df)
        scaled_prediction = make_prediction(X)

        # Inverse transform the prediction
        dummy_array = np.zeros((1, X.shape[2]))
        dummy_array[0, 3] = scaled_prediction
        inverse_transformed = scaler.inverse_transform(last_data.reshape(1, -1) + dummy_array)
        predicted_price = inverse_transformed[0, 3] - offset  # Apply the offset correction

        actual_price = df['close'].iloc[-1]
        accuracy = calculate_accuracy(actual_price, predicted_price)
        
        # Predict next price direction
        next_price_direction = "rise" if predicted_price > actual_price else "fall"
        
        logging.info(f"Current Price: {actual_price:.5f}")
        logging.info(f"Price Next {PREDICTION_INTERVAL/60} Minute(s): {predicted_price:.5f}")
        
        
        # Track direction accuracy
        if len(predicted_prices) > 0:
            predicted_direction = "rise" if predicted_price > predicted_prices[-1] else "fall"
            actual_direction = "rise" if actual_price > actual_prices[-1] else "fall"
            
            if predicted_direction == actual_direction:
                correct_direction_predictions += 1
            total_direction_predictions += 1
            
            direction_accuracy = (correct_direction_predictions / total_direction_predictions) * 100
            
            logging.info(f"Prev Predicted Direction: {predicted_direction}")
            logging.info(f"Actual Direction: {actual_direction}")
            logging.info(f"Direction Accuracy: {direction_accuracy:.2f}%")
            
        logging.info(f"Next Price Direction: {next_price_direction}")
        #logging.info(f"Prediction Accuracy: {accuracy:.2f}%")
        
        #correct_predictions += accuracy / 100
        #total_predictions += 1
        #overall_accuracy = (correct_predictions / total_predictions) * 100
        #logging.info(f"Overall Price Accuracy: {overall_accuracy:.2f}%")
        logging.info("-" * 60)

        # Append actual and predicted prices to lists
        actual_prices.append(actual_price)
        predicted_prices.append(predicted_price)

        # Plotting
        if len(actual_prices) > 1:
            plt.clf()  # Clear the current figure
            plt.plot(actual_prices, label='Actual', marker='o', linestyle='-')
            plt.plot(predicted_prices, label='Predicted', marker='o', linestyle='-')
            plt.xlabel('Time (Minutes)')
            plt.ylabel('Price')
            plt.title('Actual vs. Predicted Prices')
            plt.legend()
            plt.pause(0.1)  # Pause to update the plot

        time.sleep(PREDICTION_INTERVAL)

if __name__ == "__main__":
    main()