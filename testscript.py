import robin_stocks as r
import pandas as pd
import pandas_ta as ta
from sklearn.linear_model import LinearRegression
# async
import asyncio
# Login to Robinhood
r.login('username','password')

# Define the cryptocurrency to trade
crypto = 'BTC'

# Define the buy and sell thresholds



# Get historical data
data = pd.DataFrame(r.crypto.get_crypto_historicals(crypto, interval='5minute', span='day'))

# Calculate indicators
data.ta.ema(close='close', length=20, append=True)
data.ta.rsi(close='close', length=14, append=True)
data.ta.stoch(high='high', low='low', close='close', append=True)
data.ta.macd(close='close', append=True)
data.ta.willr(high='high', low='low', close='close', append=True)
data.ta.ichimoku(high='high', low='low', close='close', append=True)
data.ta.atr(high='high', low='low', close='close', append=True)
# calculate the linear regression
data.ta.alma(close='close', append=True)
# calculate the indicator cti or stochastic rsi
data.ta.cti(close='close', append=True)
# calculate the indicator cci or commodity channel index
data.ta.stc(close='close', append=True)

# Define buy and sell points
data['buy_points'] = calculate_buy_points(data, oversold_threshold=30)
data['sell_points'] = calculate_sell_points(data, overbought_threshold=70)

# Define stop loss
stop_loss_pct = 0.03
data['peak'] = data['close'].cummax()
data['stop_loss'] = data['peak'] * (1 - stop_loss_pct)

# Define final buy and sell signals
data['buy'] = calculate_buy_signals(data, buy_threshold=7000)
data['sell'] = calculate_sell_signals(data)

# Execute trades
for i in range(1, len(data)):
    # If we have a buy signal, then buy
    if data['buy'][i]:
        r.order_buy_crypto_limit(crypto, 1, data['close'][i])
    # If we have a sell signal, then sell
    elif data['sell'][i]:
        r.order_sell_crypto_limit(crypto, 1, data['close'][i])
