# Forex Price Prediction with LSTM

This project implements a Forex price prediction model using Long Short-Term Memory (LSTM) neural networks. It consists of two main scripts: `train.py` for training the model and `eval.py` for real-time price predictions.

## Requirements

- Python 3.7+
- MetaTrader5
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

You can install the required packages using:

```
pip install MetaTrader5 torch pandas numpy scikit-learn matplotlib joblib
```

## Training the Model (train.py)

The `train.py` script fetches historical Forex data, preprocesses it, trains an LSTM model, and saves the trained model and scaler.

### Features:
- Fetches historical data from MetaTrader5
- Adds technical indicators (EMA, MACD, RSI, Bollinger Bands, ATR, CCI)
- Implements an LSTM model with dropout for better generalization
- Uses early stopping to prevent overfitting
- Saves the best model and scaler for later use

### Usage:
```
python train.py
```

### Configuration:
You can modify the following parameters in the script:
- `SYMBOL`: The Forex pair to train on (default: "EURUSD")
- `TIMEFRAME`: The timeframe for data (default: mt5.TIMEFRAME_M1)
- `NUM_CANDLES`: Number of candles to fetch (default: 10000)
- `SEQ_LENGTH`: Sequence length for LSTM input (default: 60)
- `BATCH_SIZE`: Batch size for training (default: 16)
- `EPOCHS`: Maximum number of training epochs (default: 100)
- `LEARNING_RATE`: Learning rate for the optimizer (default: 0.001)

## Real-time Evaluation (eval.py)

The `eval.py` script loads the trained model and makes real-time predictions on live Forex data.

### Features:
- Fetches live data from MetaTrader5
- Uses the trained model to make price predictions
- Calculates prediction accuracy and direction accuracy
- Provides real-time plotting of actual vs predicted prices

### Usage:
```
python eval.py
```

### Configuration:
You can modify the following parameters in the script:
- `SYMBOL`: The Forex pair to evaluate (default: "EURUSD")
- `TIMEFRAME`: The timeframe for data (default: mt5.TIMEFRAME_M1)
- `SEQ_LENGTH`: Sequence length for LSTM input (default: 30)
- `PREDICTION_INTERVAL`: Time between predictions in seconds (default: 60)

## Output

Both scripts provide logging information and generate plots:

- `train.py` creates:
  - `training_validation_loss.png`: Plot of training and validation loss
  - `actual_vs_predicted_test.png`: Plot of actual vs predicted prices for the test set
  - `actual_vs_predicted_all.png`: Plot of actual vs predicted prices for the entire dataset

- `eval.py` provides:
  - Real-time console output with current price, predicted price, and accuracy metrics
  - Live-updating plot of actual vs predicted prices

## Notes

- Ensure that MetaTrader5 is installed and running on your system.
- The model's performance can vary depending on market conditions and the specific Forex pair being analyzed.
- This project is for educational purposes only and should not be used for actual trading without thorough testing and risk management strategies.

## License

[MIT License](https://opensource.org/licenses/MIT)
