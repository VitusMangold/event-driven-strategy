import pandas as pd
import numpy as np
import yfinance as yf

# 1. Download historical data
ticker = "AAPL"  # Example: Apple
data = yf.download(ticker, start="2020-01-01", end="2023-10-01")
print(data.head())

# 2. Add manual EPS data
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

# Forward-fill EPS data to use until the next earnings report
data['EPS_Estimate'].ffill(inplace=True)
data['Actual_EPS'].ffill(inplace=True)

# 3. Calculate RSI
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

# 5. Calculate Composite Score
weights = {
    'Volatility_Score': 0.3,
    'Volume_Score': 0.15,
    'Momentum_Score': 0.2,
    'RSI_Volatility_Score': 0.2,
    'Beta_Score': 0.15
}

# Example: Calculate Volatility_Score, Volume_Score, Momentum_Score, RSI_Volatility_Score, Beta_Score
data['Volatility_Score'] = data['Volatility'] / data['Volatility'].max()
data['Volume_Score'] = data['Volume'] / data['Volume'].max()
data['Momentum_Score'] = data['Close'].pct_change(periods=14)  # Momentum over 14 days
data['RSI_Volatility_Score'] = data['RSI'] / data['RSI'].max()
data['Beta_Score'] = np.random.uniform(0.5, 2.0, size=len(data))  # Random Beta values

def calculate_composite_score(row, weights):
    composite_score = 0
    for key, weight in weights.items():
        composite_score += row[key] * weight
    return composite_score

data['Composite_Score']= data.apply(calculate_composite_score, weights=weights, axis=1)

# 6. Check Buy Signals
def check_buy_signals(df):
    buy_signals= []

    for i in range(len(df)-3): # We need 3 days for analysis
        # 1. EPS >5% above estimates
        eps_condition= df.iloc[i]['Actual_EPS']> df.iloc[i]['EPS_Estimate']*1.05

        # 2. RSI increases by10% within3 days
        rsi_condition= (df.iloc[i+3]['RSI']- df.iloc[i]['RSI'])/ df.iloc[i]['RSI']>=0.10

        # 3. Price rises >2% next day & stays positive for2 more days
        price_condition_1= (df.iloc[i+1]['Close']- df.iloc[i]['Close'])/ df.iloc[i]['Close']>0.02
        price_condition_2= df.iloc[i+2]['Close']> df.iloc[i+1]['Close']
        price_condition_3= df.iloc[i+3]['Close']> df.iloc[i+2]['Close']

        # 4. Volatility spike on the following day
        volatility_condition= df.iloc[i+1]['Volatility']> df.iloc[i]['Volatility']*1.2 # Example:20% increase

        # All conditions must be met
        if eps_condition and rsi_condition and price_condition_1 and price_condition_2 and price_condition_3 and volatility_condition:
            buy_signals.append(df.index[i]) # Save the date of the Buy signal

    return buy_signals

buy_signals= check_buy_signals(data)
print("Buy signals on the following days:", buy_signals)

# 7. Check Sell Signals
def check_sell_signals(df):
    sell_signals= []

    for i in range(len(df)-3): # We need3 days for analysis
        # 1. EPS >5% below estimates
        eps_condition= df.iloc[i]['Actual_EPS']< df.iloc[i]['EPS_Estimate']*0.95

        # 2. RSI drops by10% within3 days
        rsi_condition= (df.iloc[i]['RSI']- df.iloc[i+3]['RSI'])/ df.iloc[i]['RSI']>=0.10

        # 3. Price falls >2% next day & stays negative for2 more days
        price_condition_1= (df.iloc[i]['Close']- df.iloc[i+1]['Close'])/ df.iloc[i]['Close']>0.02
        price_condition_2= df.iloc[i+2]['Close']< df.iloc[i+1]['Close']
        price_condition_3= df.iloc[i+3]['Close']< df.iloc[i+2]['Close']

        # 4. Volatility spike on the following day
        volatility_condition= df.iloc[i+1]['Volatility']> df.iloc[i]['Volatility']*1.2 # Example:20% increase

        # All conditions must be met
        if eps_condition and rsi_condition and price_condition_1 and price_condition_2 and price_condition_3 and volatility_condition:
            sell_signals.append(df.index[i]) # Save the date of the Sell signal

    return sell_signals

sell_signals= check_sell_signals(data)
print("Sell signals on the following days:", sell_signals)

# 8. Exit conditions for Buy positions
def check_buy_exit(df, entry_index):
    exit_conditions= []

    for i in range(entry_index+1, min(entry_index+4, len(df))): # Check the next3 days
        # 1. Price drops >2%
        price_drop_condition= (df.iloc[entry_index]['Close']- df.iloc[i]['Close'])/ df.iloc[entry_index]['Close']>0.02

        # 2. RSI falls >5%
        rsi_drop_condition= (df.iloc[entry_index]['RSI']- df.iloc[i]['RSI'])/ df.iloc[entry_index]['RSI']>0.05

        # If either condition is met, save the exit date
        if price_drop_condition or rsi_drop_condition:
            exit_conditions.append(df.index[i])
            break # Stop the loop once a condition is met

    return exit_conditions

# 9. Exit conditions for Sell positions
def check_sell_exit(df, entry_index):
    exit_conditions= []

    for i in range(entry_index+1, min(entry_index+4, len(df))): # Check the next3 days
        # 1. Price rises >2%
        price_rise_condition= (df.iloc[i]['Close']- df.iloc[entry_index]['Close'])/ df.iloc[entry_index]['Close']>0.02

        # 2. RSI increases >5%
        rsi_rise_condition= (df.iloc[i]['RSI']- df.iloc[entry_index]['RSI'])/ df.iloc[entry_index]['RSI']>0.05

        # If either condition is met, save the exit date
        if price_rise_condition or rsi_rise_condition:
            exit_conditions.append(df.index[i])
            break # Stop the loop once a condition is met

    return exit_conditions

# 10. Trading strategy
def trading_strategy(df):
    signals= [] # Stores all signals (Buy, Sell, Exit)
    position= None # Current position (None, "Buy", "Sell")

    for i in range(len(df)-3): # We need3 days for analysis
        # Check Buy signal
        if check_buy_signals(df.iloc[i:i+4].reset_index(drop=True)): # Check the next3 days
            signals.append((df.index[i], "Buy"))
            position= "Buy"

        # Check Sell signal
        elif check_sell_signals(df.iloc[i:i+4].reset_index(drop=True)):
            signals.append((df.index[i], "Sell"))
            position= "Sell"

        # Check exit conditions if a position is open
        if position== "Buy":
            exit_dates= check_buy_exit(df, i)
            if exit_dates:
                signals.append((exit_dates[0], "Exit Buy"))
                position= None
        elif position== "Sell":
            exit_dates= check_sell_exit(df, i)
            if exit_dates:
                signals.append((exit_dates[0], "Exit Sell"))
                position= None

    return signals

# Apply the strategy to the data
signals= trading_strategy(data)
print("Signals:", signals)