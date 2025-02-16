import pandas as pd
import numpy as np
import yfinance as yf

# 1. Download historical stock data
ticker = "AAPL"  # Example: Apple
data = yf.download(ticker, start="2020-01-01", end="2023-10-01")
print(data.head())

# 2. Add manually defined EPS data (Earnings Per Share)
eps_data = {
    '2020-01-28': {'Actual_EPS': 4.99, 'EPS_Estimate': 4.54},
    '2020-04-30': {'Actual_EPS': 2.55, 'EPS_Estimate': 2.26},
    '2020-07-30': {'Actual_EPS': 2.58, 'EPS_Estimate': 2.07},
    '2020-10-29': {'Actual_EPS': 0.73, 'EPS_Estimate': 0.70},
    '2021-01-27': {'Actual_EPS': 1.68, 'EPS_Estimate': 1.41},
    '2021-04-28': {'Actual_EPS': 1.40, 'EPS_Estimate': 0.99},
    '2021-07-27': {'Actual_EPS': 1.30, 'EPS_Estimate': 1.01},
    '2021-10-28': {'Actual_EPS': 1.24, 'EPS_Estimate': 1.24},
    '2022-01-27': {'Actual_EPS': 2.10, 'EPS_Estimate': 1.89},
    '2022-04-28': {'Actual_EPS': 1.52, 'EPS_Estimate': 1.43},
    '2022-07-28': {'Actual_EPS': 1.20, 'EPS_Estimate': 1.16},
    '2022-10-27': {'Actual_EPS': 1.29, 'EPS_Estimate': 1.27},
    '2023-02-02': {'Actual_EPS': 1.88, 'EPS_Estimate': 1.94},
    '2023-05-04': {'Actual_EPS': 1.52, 'EPS_Estimate': 1.43},
    '2023-08-03': {'Actual_EPS': 1.26, 'EPS_Estimate': 1.19},
}

# Add EPS data to the DataFrame
data['EPS_Estimate'] = np.nan
data['Actual_EPS'] = np.nan

for date, eps_values in eps_data.items():
    if date in data.index:
        data.at[date, 'EPS_Estimate'] = eps_values['EPS_Estimate']
        data.at[date, 'Actual_EPS'] = eps_values['Actual_EPS']

# Forward-fill EPS data so that the most recent values are used until the next earnings report
data['EPS_Estimate'].ffill(inplace=True)
data['Actual_EPS'].ffill(inplace=True)

# 3. Calculate RSI (Relative Strength Index)
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data)

# 4. Calculate volatility (standard deviation of closing prices over 14 days)
data['Volatility'] = data['Close'].rolling(window=14).std()

# 5. Define Buy Signal Conditions
def check_buy_signals(df):
    buy_signals = []

    for i in range(len(df)-3):  # We need at least 3 days of analysis
        # 1. Actual EPS is more than 5% above the estimated EPS
        eps_condition = df.iloc[i]['Actual_EPS'] > df.iloc[i]['EPS_Estimate'] * 1.05

        # 2. RSI increases by at least 10% within 3 days
        rsi_condition = (df.iloc[i+3]['RSI'] - df.iloc[i]['RSI']) / df.iloc[i]['RSI'] >= 0.10

        # 3. Price increases by more than 2% the next day and continues to rise for two more days
        price_condition_1 = (df.iloc[i+1]['Close'] - df.iloc[i]['Close']) / df.iloc[i]['Close'] > 0.02
        price_condition_2 = df.iloc[i+2]['Close'] > df.iloc[i+1]['Close']
        price_condition_3 = df.iloc[i+3]['Close'] > df.iloc[i+2]['Close']

        # 4. Volatility spike on the following day (at least 20% increase)
        volatility_condition = df.iloc[i+1]['Volatility'] > df.iloc[i]['Volatility'] * 1.2

        # All conditions must be met to generate a Buy signal
        if eps_condition and rsi_condition and price_condition_1 and price_condition_2 and price_condition_3 and volatility_condition:
            buy_signals.append(df.index[i])

    return buy_signals

# Apply Buy Signal Check
buy_signals = check_buy_signals(data)
print("Buy signals on the following days:", buy_signals)

# 6. Define Sell Signal Conditions
def check_sell_signals(df):
    sell_signals = []

    for i in range(len(df)-3):  # We need at least 3 days of analysis
        # 1. Actual EPS is more than 5% below the estimated EPS
        eps_condition = df.iloc[i]['Actual_EPS'] < df.iloc[i]['EPS_Estimate'] * 0.95

        # 2. RSI drops by at least 10% within 3 days
        rsi_condition = (df.iloc[i]['RSI'] - df.iloc[i+3]['RSI']) / df.iloc[i]['RSI'] >= 0.10

        # 3. Price decreases by more than 2% the next day and continues to fall for two more days
        price_condition_1 = (df.iloc[i]['Close'] - df.iloc[i+1]['Close']) / df.iloc[i]['Close'] > 0.02
        price_condition_2 = df.iloc[i+2]['Close'] < df.iloc[i+1]['Close']
        price_condition_3 = df.iloc[i+3]['Close'] < df.iloc[i+2]['Close']

        # 4. Volatility spike on the following day (at least 20% increase)
        volatility_condition = df.iloc[i+1]['Volatility'] > df.iloc[i]['Volatility'] * 1.2

        # All conditions must be met to generate a Sell signal
        if eps_condition and rsi_condition and price_condition_1 and price_condition_2 and price_condition_3 and volatility_condition:
            sell_signals.append(df.index[i])

    return sell_signals

# Apply Sell Signal Check
sell_signals = check_sell_signals(data)
print("Sell signals on the following days:", sell_signals)

# 7. Backtesting Function
def backtest(df, signals, initial_balance=10000):
    balance = initial_balance
    position = None
    entry_price = 0

    for date, signal in signals:
        price = df.loc[date, 'Close']

        if signal == "Buy" and position is None:
            position = balance / price  # Buy as many shares as possible
            entry_price = price
            balance = 0  # All money invested

        elif signal == "Sell" and position is not None:
            balance = position * price  # Sell all shares
            position = None

    # If still holding a position at the end, sell it
    if position is not None:
        balance = position * df.iloc[-1]['Close']

    return {
        "Final Balance": balance,
        "Return (%)": ((balance - initial_balance) / initial_balance) * 100,
    }

# Convert buy signals and sell signals into tuples with labels
signals = [(date, "Buy") for date in buy_signals] + [(date, "Sell") for date in sell_signals]

# Run Backtesting
results = backtest(data, signals)

print(results)
