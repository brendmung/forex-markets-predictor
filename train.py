import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration parameters
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
START_POS = 0
NUM_CANDLES = 10000
SEQ_LENGTH = 60
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_ROUNDS = 10
CSV_FILE = "forex_training_data.csv"

def fetch_and_save_data():
    # Connect to MetaTrader 5
    if not mt5.initialize():
        logging.error("MT5 initialization failed")
        mt5.shutdown()
        exit()

    # Fetch historical data
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, START_POS, NUM_CANDLES)
    if rates is None:
        logging.error("Failed to fetch data")
        mt5.shutdown()
        exit()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Add technical indicators
    df = add_technical_indicators(df)
    df = df.dropna()  # Remove NaN values resulting from indicator calculation

    # Save to CSV
    df.to_csv(CSV_FILE, index=False)
    logging.info(f"Data saved to {CSV_FILE}")

    mt5.shutdown()
    return df

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

# Check if CSV exists, if not, fetch and save data
if not os.path.exists(CSV_FILE):
    df = fetch_and_save_data()
else:
    df = pd.read_csv(CSV_FILE)
    df['time'] = pd.to_datetime(df['time'])
    logging.info(f"Data loaded from {CSV_FILE}")

# Prepare the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'tick_volume', 
                                       'EMA10', 'EMA30', 'MACD', 'Signal_Line', 'RSI', 
                                       'BB_upper', 'BB_lower', 'ATR', 'CCI']].values)

X, y = [], []
for i in range(len(scaled_data) - SEQ_LENGTH):
    X.append(scaled_data[i:i+SEQ_LENGTH])
    y.append(scaled_data[i+SEQ_LENGTH, 3])

X, y = np.array(X), np.array(y)

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Define the LSTM model with dropout
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

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ForexLSTM(input_dim=14, hidden_dim=28, num_layers=2, output_dim=1).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Training loop with early stopping
best_loss = float('inf')
counter = 0
train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for i in range(0, len(X_train), BATCH_SIZE):
        batch_X = torch.FloatTensor(X_train[i:i+BATCH_SIZE]).to(device)
        batch_y = torch.FloatTensor(y_train[i:i+BATCH_SIZE]).view(-1, 1).to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    epoch_loss /= len(X_train) // BATCH_SIZE
    train_losses.append(epoch_loss)
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()
        val_loss = mean_squared_error(y_val, val_outputs)
        val_losses.append(val_loss)
    
    logging.info(f"Epoch [{epoch}/{EPOCHS}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_forex_model.pth')
        joblib.dump(scaler, 'best_forex_scaler.joblib')
    else:
        counter += 1
    
    if counter >= EARLY_STOPPING_ROUNDS:
        logging.info(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(torch.load('best_forex_model.pth'))

model.eval()
with torch.no_grad():
    test_predictions = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

num_features = X_test.shape[2]
test_predictions_reshaped = np.zeros((test_predictions.shape[0], num_features))
test_predictions_reshaped[:, 3] = test_predictions.flatten()

last_time_step = X_test[:, -1, :]
inverse_transformed_predictions = scaler.inverse_transform(last_time_step + test_predictions_reshaped)
test_predictions = inverse_transformed_predictions[:, 3]

y_test_reshaped = np.zeros((y_test.shape[0], num_features))
y_test_reshaped[:, 3] = y_test
inverse_transformed_actual = scaler.inverse_transform(last_time_step + y_test_reshaped)
y_test_inverse = inverse_transformed_actual[:, 3]

test_rmse = np.sqrt(mean_squared_error(y_test_inverse, test_predictions))
logging.info(f"Test RMSE: {test_rmse}")

# Calculate accuracy for test set
def calculate_accuracy(y_true, y_pred, threshold=0.01):
    correct_predictions = np.abs(y_true - y_pred) <= threshold * y_true
    accuracy = np.mean(correct_predictions) * 100
    return accuracy

test_accuracy = calculate_accuracy(y_test_inverse, test_predictions)
logging.info(f"Test Accuracy (within 1% threshold): {test_accuracy:.2f}%")

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss', marker='o', linestyle='-', color='blue', alpha=0.7)
plt.plot(val_losses, label='Validation Loss', marker='x', linestyle='--', color='red', alpha=0.7)
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_validation_loss.png')
plt.show()

# Plot actual vs predicted prices for test set
plt.figure(figsize=(12, 6))
plt.plot(y_test_inverse, label='Actual Price', marker='o', linestyle='-', color='blue', alpha=0.7)
plt.plot(test_predictions, label='Predicted Price', marker='x', linestyle='--', color='red', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted Prices (Test Set)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.savefig('actual_vs_predicted_test.png')
plt.show()

# Evaluate on all data
with torch.no_grad():
    all_predictions = model(torch.FloatTensor(X).to(device)).cpu().numpy()

all_predictions_reshaped = np.zeros((all_predictions.shape[0], num_features))
all_predictions_reshaped[:, 3] = all_predictions.flatten()

last_time_step = X[:, -1, :]
inverse_transformed_predictions = scaler.inverse_transform(last_time_step + all_predictions_reshaped)
all_predictions = inverse_transformed_predictions[:, 3]

y_reshaped = np.zeros((y.shape[0], num_features))
y_reshaped[:, 3] = y
inverse_transformed_actual = scaler.inverse_transform(last_time_step + y_reshaped)
y_inverse = inverse_transformed_actual[:, 3]

overall_rmse = np.sqrt(mean_squared_error(y_inverse, all_predictions))
average_accuracy = calculate_accuracy(y_inverse, all_predictions)

logging.info(f"Overall RMSE: {overall_rmse}")
logging.info(f"Average Accuracy (within 1% threshold): {average_accuracy:.2f}%")

# Plot actual vs predicted prices for the entire dataset
plt.figure(figsize=(12, 6))
plt.plot(y_inverse, label='Actual Price', marker='o', linestyle='-', color='blue', alpha=0.7)
plt.plot(all_predictions, label='Predicted Price', marker='x', linestyle='-', color='red', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted Prices (Entire Dataset)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.savefig('actual_vs_predicted_all.png')
plt.show()


torch.save(model.state_dict(), 'forex_model.pth')
joblib.dump(scaler, 'forex_scaler.joblib')

logging.info("Model and scaler saved successfully.")

print(f"Test RMSE: {test_rmse}")
print(f"Test Accuracy (within 1% threshold): {test_accuracy:.2f}%")
print(f"Overall RMSE: {overall_rmse}")
print(f"Average Accuracy (within 1% threshold): {average_accuracy:.2f}%")