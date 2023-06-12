# %%
from robin_stocks import robinhood as r
import pandas as pd
from datetime import datetime
import logging
# import traceback # for debugging
from sys import exit # for debugging
import traceback # for debugging
import time
from os import path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from tqdm import tqdm
import numpy as np
import json
from icecream import ic
from colorama import Fore, Back, Style
# Login to Robinhood
with open('secrets.json'):
    username = secrets['username']
    password = secrets['password']
r.login(username=username,
        password=password)
panic_mode = False # set to True to sell all crypto positions immediately
# List of cryptocurrencies to trade
cryptos = ['BTC', 'ETH', 'ADA', 'DOGE', 'MATIC', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'SOL', 'AVAX', 'XLM', 'BCH', 'XTZ']

# log file
logger = logging.getLogger('crypto_trader')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('crypto_trader.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
# import datetime now
from datetime import datetime
# import now function
from datetime import datetime
# import time
import time

now = datetime.now()

# Initialize an empty dataframe to store crypto data
crypto_data = pd.DataFrame()

def get_crypto_data(cryptos, crypto_data):

    for crypto in tqdm(cryptos):
        try:
            #print(f'Getting data for {crypto}, please wait...',end='')
            # Get historical data
            historicals = r.get_crypto_historicals(crypto, interval='day', span='week')
            current_price = historicals[-1]['close_price']
            historical_prices = [x['close_price'] for x in historicals]

            # Load account profile
            profile = r.profiles.load_account_profile()
            buying_power = profile['buying_power']

            # Get crypto positions
            positions = r.crypto.get_crypto_positions()
            print(f'found {len(positions)} positions.')
            for position in tqdm(positions):
                # print(position)
                if position['currency']['code'] == crypto:
                    pos_dict = position['currency']
                    min_order_size = float(pos_dict['increment'])
                    coin_holdings = float(position['quantity_available'])


            # Get profile data
            profile_data = r.profiles.load_portfolio_profile()
            current_equity = profile_data['equity']
            current_equity = float(current_equity)
            previous_equity = profile_data['adjusted_equity_previous_close']
            previous_equity = float(previous_equity)
            daily_profit = float(current_equity) - float(previous_equity)

            # convert all the numerical values in historical_prices to float
            historical_prices = [float(x) for x in historical_prices]

            # Append data to dataframe
            crypto_data = crypto_data.append({
                'ticker': crypto,
                'current_price': current_price,
                'historical_prices': historical_prices,
                'buying_power': buying_power,
                'min_order_size': min_order_size,
                'coin_holdings': coin_holdings,
                'updated_at': now,
                'current_equity': current_equity,
                'previous_equity': previous_equity,
                'daily_profit': daily_profit
            }, ignore_index=True)
        except Exception as e:
            print(f'Error getting data for {crypto}: {e}')
            logger.error(f'Error getting data for {crypto}: {e}')
            traceback.print_exc()
            pass

    def add_me(crypto_data, metric, metric_name):
        if isinstance(metric, pd.Series):
            # if rsi is a series, get the last value
            crypto_data.loc[index, metric_name] = metric.iloc[-1]
        elif isinstance(metric, float):
            # if rsi is a float, just use it
            crypto_data.loc[index, metric_name] = metric
        elif isinstance(metric, np.ndarray):
            # if rsi is an array, get the last value
            crypto_data.loc[index, metric_name] = metric[-1]
        else:
            # otherwise, just use the value
            crypto_data.loc[index, metric_name] = metric
        return crypto_data

    # Calculate RSI, MACD, Bollinger Bands, etc. and add to dataframe
    for index, row in tqdm(crypto_data.iterrows()):
        prices = pd.Series(row['historical_prices'])

        # Calculate RSI (Relative Strength Index)
        # first make a copy of the data frame twice
        up_df, down_df = prices.copy(), prices.copy()
        # For up days, if the price is lower than previous price, set price to 0.
        up_df[up_df < up_df.shift(1)] = 0
        # For down days, if the price is higher than previous price, set price to 0.
        down_df[down_df > down_df.shift(1)] = 0
        # We need change and average gain for the first calculation.
        # Calculate the 1st-day change and average gain
        up_df.iloc[0] = 0
        down_df.iloc[0] = 0
        # Calculate the EWMA (Exponential Weighted Moving Average)
        roll_up1 = up_df.ewm(span=14).mean()
        roll_down1 = down_df.ewm(span=14).mean()
        # Calculate the RSI based on EWMA
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))
        rsi = RSI1 # save it for later

        # Calculate MACD (Moving Average Convergence Divergence)
        exp1 = prices.ewm(span=12, adjust=False).mean()
        # Calculate the 26-day EMA
        exp2 = prices.ewm(span=26, adjust=False).mean()
        # Subtract 26-day EMA from 12-day EMA, and you will get the first MACD
        macd = exp1 - exp2
        # Calculate the 9-day EMA of the MACD above
        exp3 = macd.ewm(span=9, adjust=False).mean()
        # Plot the MACD and exp3
        macd = macd - exp3
        # save it for later
        macd = macd.iloc[-1]
        macd_signal = exp3.iloc[-1] # the signal line
        macd_hist = macd - macd_signal # the histogram
        ma200 = prices.rolling(200).mean().iloc[-1] # 200 day moving average
        ma50 = prices.rolling(50).mean().iloc[-1] # 50 day moving average
        ma20 = prices.rolling(20).mean().iloc[-1] # 20 day moving average
        sma = prices.rolling(5).mean().iloc[-1] # 5 day simple moving average
        ema = prices.ewm(span=5, adjust=False).mean().iloc[-1] # 5 day exponential moving average

        # Calculate Bollinger Bands
        # Calculating the Upper and Lower Bands
        # We will be using a 14 day window
        window = 14
        # Calculate rolling mean and standard deviation using number of days set above
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        # Create two new DataFrame columns to hold values of upper and lower Bollinger bands
        upper_band = pd.Series()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)

        # Add to dataframe
        print(f'Adding technical indicators for {row["ticker"]} to dataframe\n\t...')
        #^ Add the RSI
        crypto_data = add_me(crypto_data, rsi, 'rsi')
        #^ Add the MACD
        crypto_data = add_me(crypto_data, macd, 'macd')
        #^ Add the SMA from the Bollinger Bands
        crypto_data = add_me(crypto_data, rolling_mean, 'sma')
        #^ Add the EMA from the Bollinger Bands
        crypto_data = add_me(crypto_data, ema, 'ema')
        #^ Add the 200 day moving average
        crypto_data = add_me(crypto_data, ma200, 'ma200')
        #^ Add the 50 day moving average
        crypto_data = add_me(crypto_data, ma50, 'ma50')
        #^ Add the 20 day moving average
        crypto_data = add_me(crypto_data, ma20, 'ma20')
        #^ Add the 5 day simple moving average
        crypto_data = add_me(crypto_data, sma, 'sma')
        #^ Add the 5 day exponential moving average
        crypto_data = add_me(crypto_data, ema, 'ema')
        #^ Add the MACD signal line
        crypto_data = add_me(crypto_data, macd_signal, 'macd_signal')
        #^ Add the MACD histogram
        crypto_data = add_me(crypto_data, macd_hist, 'macd_hist')
        #^ Add the Upper Band
        crypto_data = add_me(crypto_data, upper_band, 'upper_band')
        #^ Add the Lower Band
        crypto_data = add_me(crypto_data, lower_band, 'lower_band')
    # save the crypto_data dataframe to a csv file
    crypto_data.to_csv('crypto_data.csv', index=False)
    return crypto_data


# %%
from colorama import Fore, Back, Style
color_options = [
    Fore.BLACK,
    Fore.RED,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.BLUE,
    Fore.MAGENTA,
    Fore.CYAN,
    Fore.WHITE,
    Fore.LIGHTBLACK_EX,
    Fore.LIGHTRED_EX,
    Fore.LIGHTGREEN_EX,
    Fore.LIGHTYELLOW_EX,
    Fore.LIGHTBLUE_EX,
    Fore.LIGHTMAGENTA_EX,
    Fore.LIGHTCYAN_EX
]
background_options = [
    Back.BLACK,
    Back.RED,
    Back.GREEN,
    Back.YELLOW,
    Back.BLUE,
    Back.MAGENTA,
    Back.CYAN,
    Back.WHITE,
    Back.LIGHTBLACK_EX,
    Back.LIGHTRED_EX,
    Back.LIGHTGREEN_EX,
    Back.LIGHTYELLOW_EX,
    Back.LIGHTBLUE_EX,
    Back.LIGHTMAGENTA_EX,
    Back.LIGHTCYAN_EX
]
style_options = [
    Style.DIM,
    Style.NORMAL,
    Style.BRIGHT,
]

# %% [markdown]
# def update_signals():
#     stop_loss = 0.95 # 5% loss
#     take_profit = 1.03 # 3% profit
#     try:
#
#
#         # Generate trading signals based on technical indicators
#         for index, row in tqdm(crypto_data.iterrows()):
#             # Convert historical_prices to a list if it's not already
#             if isinstance(row['historical_prices'], str):
#                 historical_prices = [float(price) for price in row['historical_prices'].strip('[]').split(',')]
#             else:
#                 historical_prices = [float(x) for x in row['historical_prices']]
#
#             # convert all values to floats in the row that are numbers
#             for col in crypto_data.columns[1:]:
#                 try:
#                     # drop NaN values
#                     if row[col] == 'NaN':
#                         crypto_data.loc[index, col] = np.nan
#                     else:
#                         crypto_data.loc[index, col] = float(row[col])
#                 except:
#                     pass
#
#             historical_prices_last_5 = historical_prices[-5:]
#             # Calculate the percent change over the last 5 hours
#             if all(price != 0.0 for price in historical_prices_last_5):
#                 # if all the values are not zero, calculate the percent change
#                 pct_change_5h = (float(row['current_price']) - historical_prices_last_5[-1]) / historical_prices_last_5[-1]
#             else:
#                 # calculate the percent change from the last non-zero value
#                 try:
#                     for price in historical_prices_last_5:
#                         if price != 0.0:
#                             pct_change_5h = (float(row['current_price']) - price) / price
#                             break
#                 except:
#                     pct_change_5h = 0.0  # or handle the zero values differently
#
#             # ─── Signal Calculations ──────────────────────────────
#
#
#             #print(f'pct_change_5h: {pct_change_5h}')
#             crypto_data.loc[index, 'signal'] = 0  # set the default value of signal to 0
#             # Check the conditions for each signal
#             if (float(row['rsi']) < 30) and (float(row['macd']) > 0):  # if the rsi dips below 30 indicating oversold and the macd is positive, buy
#                 # if the signal is a buy background is green and text should be black
#                 print(Fore.BLACK + f'BUY SIGNAL: {row["ticker"]}' + Fore.RESET)
#                 crypto_data.loc[index, 'signal'] +=1
#             elif (float(row['rsi']) > 70) and (float(row['macd']) < 0):  # if the rsi goes above 70 indicating overbought and the macd is negative, sell
#                 print(Fore.GREEN + f'SELL SIGNAL: {row["ticker"]}' + Fore.RESET)
#                 crypto_data.loc[index, 'signal'] -= -1
#             elif (float(row['macd']) > float(row['macd_signal'])) and (float(row['current_price']) > float(row['sma'])):  # if the macd line crosses above the signal line and the price is above the SMA, buy
#                 print(Fore.YELLOW + f'BUY SIGNAL: {row["ticker"]}' + Fore.RESET)
#                 crypto_data.loc[index, 'signal'] += 1
#             elif (float(row['macd']) < float(row['macd_signal'])) and (float(row['current_price']) < float(row['sma'])):  # if the macd line crosses below the signal line and the price is below the SMA, sell
#                 print(Fore.RED + f'SELL SIGNAL: {row["ticker"]}' + Fore.RESET)
#                 crypto_data.loc[index, 'signal'] += -1
#             elif (float(row['current_price']) > float(row['ma200'])):  # if the price crosses above the 200-day moving average, buy
#                 print(Fore.BLUE + f'BUY SIGNAL: {row["ticker"]}' + Fore.RESET)
#                 crypto_data.loc[index, 'signal'] += 1
#             elif (float(row['current_price']) < float(row['ma200'])):  # if the price crosses below the 200-day moving average, sell
#                 print(Back.GREEN + f'SELL SIGNAL: {row["ticker"]}' + Fore.RESET)
#                 crypto_data.loc[index, 'signal'] += -1
#             elif (pct_change_5h > 0.05):  # if the price has increased by more than 5% in the last 5 hours, buy
#                 print(Back.MAGENTA + f'BUY SIGNAL: {row["ticker"]}' + Fore.RESET)
#                 crypto_data.loc[index, 'signal'] += 1
#             elif (pct_change_5h < -0.05):  # if the price has decreased by more than 5% in the last 5 hours, sell
#                 print(Fore.LIGHTMAGENTA_EX + f'SELL SIGNAL: {row["ticker"]}' + Fore.RESET)
#                 crypto_data.loc[index, 'signal'] += -1
#             else:
#                 crypto_data.loc[index, 'signal'] = 0
#
#             # consider limit loss and take profit here
#             # if the value of the coin has risen by the take profit percentage, sell
#             if (float(row['current_price']) / float(row['buy_price'])) > take_profit:
#                 print(Back.BLACK + Fore.GREEN + f'TAKE PROFIT: {row["ticker"]}' + Fore.RESET)
#                 crypto_data.loc[index, 'signal'] -= -1
#             elif (float(row['current_price']) / float(row['buy_price'])) < stop_loss:
#                 print(Back.BLACK + Fore.RED + f'STOP LOSS: {row["ticker"]}' + Fore.RESET)
#                 crypto_data.loc[index, 'signal'] -=1
#
#             # using stop_loss and take_profit variables
#         # save the crypto_data dataframe to a csv file
#         crypto_data.to_csv('crypto_data.csv', index=False)
#         return crypto_data
#     except Exception as e:
#         crypto_data.loc[index, 'signal'] = 0
#         print(f'Error: {e}')
#         return crypto_data
#

# %%
def update_signals():
    stop_loss = 0.95 # 5% loss
    take_profit = 1.03 # 3% profit
    transactions_df = pd.read_csv('transactions.csv')

    def process_prices(x):
        if isinstance(x, str):
            return [float(price) for price in x.strip('[]').split(',')]
        return [float(price) for price in x]

    def calculate_pct_change(historical_prices, current_price):
        historical_prices_last_5 = historical_prices[-5:]
        if all(price != 0.0 for price in historical_prices_last_5):
            return (current_price - historical_prices_last_5[-1]) / historical_prices_last_5[-1]
        else:
            for price in historical_prices_last_5:
                if price != 0.0:
                    return (current_price - price) / price
        return 0.0

    def calculate_signal(row):
        if row['rsi'] < 30 and row['macd'] > 0:
            print(Fore.BLACK + f'BUY SIGNAL: {row["ticker"]}' + Fore.RESET)
            return 1
        elif row['rsi'] > 70 and row['macd'] < 0:
            print(Fore.GREEN + f'SELL SIGNAL: {row["ticker"]}' + Fore.RESET)
            return -1
        elif row['macd'] > row['macd_signal'] and row['current_price'] > row['sma']:
            print(Fore.YELLOW + f'BUY SIGNAL: {row["ticker"]}' + Fore.RESET)
            return 1
        elif row['macd'] < row['macd_signal'] and row['current_price'] < row['sma']:
            print(Fore.RED + f'SELL SIGNAL: {row["ticker"]}' + Fore.RESET)
            return -1
        elif row['current_price'] > row['ma200']:
            print(Fore.BLUE + f'BUY SIGNAL: {row["ticker"]}' + Fore.RESET)
            return 1
        elif row['current_price'] < row['ma200']:
            print(Back.GREEN + f'SELL SIGNAL: {row["ticker"]}' + Fore.RESET)
            return -1
        elif row['pct_change_5h'] > 0.05:
            print(Back.MAGENTA + f'BUY SIGNAL: {row["ticker"]}' + Fore.RESET)
            return 1
        elif row['pct_change_5h'] < -0.05:
            print(Fore.LIGHTMAGENTA_EX + f'SELL SIGNAL: {row["ticker"]}' + Fore.RESET)
            return -1
        else:
            return 0

    try:
        crypto_data['historical_prices'] = crypto_data['historical_prices'].apply(process_prices)
        crypto_data.replace('NaN', np.nan, inplace=True)
        for col in crypto_data.columns[1:]:
            crypto_data[col] = pd.to_numeric(crypto_data[col], errors='coerce')

        crypto_data['pct_change_5h'] = crypto_data.apply(lambda x: calculate_pct_change(x['historical_prices'], x['current_price']), axis=1)
        crypto_data['updated_date'] = pd.to_datetime(crypto_data['updated_date'], errors='coerce').fillna(pd.to_datetime('2021-01-01'))
        crypto_data['signal'] = 0

        crypto_data['signal'] = crypto_data.apply(calculate_signal, axis=1)

        # Apply stop loss and take profit logic
        crypto_data['buy_price'] = crypto_data['ticker'].apply(lambda x: transactions_df.loc[transactions_df['ticker'] == x, 'buy_price'].iloc[-1] if x in transactions_df['ticker'].values else np.nan)
        crypto_data.loc[(crypto_data['current_price'] / crypto_data['buy_price']) > take_profit, 'signal'] = -1
        crypto_data.loc[(crypto_data['current_price'] / crypto_data['buy_price']) < stop_loss, 'signal'] = -1

        # save the crypto_data dataframe to a csv file
        crypto_data.to_csv('crypto_data.csv', index=False)
        return crypto_data
    except Exception as e:
        crypto_data['signal'] = 0
        print(f'Error: {e}')
        # log the traceback
        traceback.print_exc()

        return crypto_data


# %%

crypto_data = get_crypto_data(cryptos, crypto_data)

crypto_data.to_csv('crypto_data.csv', index=False)
# show the dataframe
crypto_data.head()

# %%
dir(r)

# %%
dir(r.orders)

# %%
import time
import datetime
from ratelimit import limits, sleep_and_retry

# @sleep_and_retry
# def sell_coin(coin, current_price, buying_power, volume, crypto_positions):
#     # use robin stocks to sell the coin
#     # if we are selling everything then cancel all other sell orders first
#     if volume == crypto_positions[coin]:
#         r.cancel_all_crypto_orders(coin)
#     # sell the coin
#     r.order_sell_crypto_by_quantity(
#         symbol=coin,
#         quantity=volume,
#         timeInForce='gtc',
#         jsonify=True
#     )
#     print(f'Selling {volume} {coin} at {current_price}...')
#     # calculate the new buying power
#     buying_power += current_price * volume
#     # return the new buying power
#     time.sleep(1)
#     return buying_power

# @sleep_and_retry
# def buy_coin(coin, current_price, buying_power, dollars_spent):
#     # use robin stocks to buy the coin
#     r.order_buy_crypto_by_price(
#         symbol=coin,
#         amountInDollars=dollars_spent,
#         timeInForce='gtc',
#         jsonify=True
#     )
#     print(f'Buying {dollars_spent} {coin} at {current_price}...')
#     # calculate the new buying power
#     buying_power -= dollars_spent
#     # return the new buying power
#     time.sleep(1)
#     return buying_power


# %%
def order_crypto(symbol, quantity_or_price, amount_in='dollars', side='buy'):
    """
    The order_crypto function is used to place a crypto order.

    :param symbol: Specify the coin you want to buy
    :param quantity_or_price: Specify the amount of crypto you want to buy or sell
    :param amount_in: Specify the amount of money you want to spend on buying a coin
    :param side: Specify whether you want to buy or sell the coin
    :return: A dictionary with the following keys:
    :doc-author: Trelent
    """

    if side == 'buy':
        amountIn = 'dollars'
    else:
        amountIn = 'quantity'

    # cast the quantity_or_price to a string
    quantity_or_price = str(quantity_or_price)

    try:
        # use robin stocks to buy the coin
        r.orders.order_crypto(
            symbol=str(symbol),
            quantityOrPrice=float(quantity_or_price),
            amountIn=str(amount_in), # either 'quantity' or 'dollars'
            side=str(side),
            timeInForce='gtc',
            jsonify=True
        )
    except Exception as e:
        raise e



# %%
# Read the crypto_data.csv file with the correct data types
crypto_data = pd.read_csv('crypto_data.csv', converters={'historical_prices': eval})
buying_power = float(r.profiles.load_account_profile(info='crypto_buying_power')) * 0.60 # only use 60% of the buying power

def is_daytime():
    # CHECK TIME OF DAY IN CENTRAL STANDARD TIME
    current_time = datetime.datetime.now()
    if current_time.hour >= 11 and current_time.hour <= 23:
        return True
    else:
        return False

print(f'I can only play with {buying_power} dollars')

def main_loop():
    """
    The main_loop function is the main function of this program. It runs every ten minutes and updates the signals for each coin, then checks if we own any coins or not. If we do, it will check to see if there are any sell signals (signal &lt; -0.5) and sell 70% of our holdings in that coin; otherwise, it will check to see if there are buy signals (signal &gt; 0.5) and buy 50% of 1% of our buying power worth in that coin.

    :return: The following:
    :doc-author: Trelent
    """
    global crypto_data

    while True:
        # cancel all orders
        r.cancel_all_crypto_orders()
        time.sleep(10)

        # CHECK TIME OF DAY IN CENTRAL STANDARD TIME
        current_time = datetime.datetime.now()

        # IF ITS AFTER 11PM AND BEFORE 5AM, THEN THE MARKET IS CLOSED SO ONLY DEAL WITH CRYPTO, ALSO ALL SIGNALS SHOULD HAVE A REDUCED BUY_VOLUME BY REDUCING BUY_IN BY 90% KEEPING SELL VOLUMES THE SAME.
        if current_time.hour >= 23 or current_time.hour < 5:
            buying_limiter = 1 # 100% of buying power during the day
            pass
        else:
            buying_limiter = 0.02 # 2% of buying power at night


        crypto_data = get_crypto_data(cryptos, crypto_data)
        try:
            # Update signals
            # Replace with your own function or code to update signals
            update_signals()

            # Get the current time
            now = datetime.datetime.now()

            for index, row in crypto_data.iterrows():
                coin = row['ticker']
                signal = row['signal']
                current_price = float(row['current_price'])
                buying_power = float(row['buying_power'])
                coin_holdings = float(row['coin_holdings'])
                try:
                    crypto_positions = row['crypto_positions']
                except:
                    crypto_positions = coin_holdings
                print(f' >> {row["ticker"]} << signal: {row["signal"]}, holding: {row["coin_holdings"]}, ${buying_power}')

                try:
                    # Check if we own the coin
                    if coin_holdings > 0:
                        pass
                    else:
                        # We do not own the coin
                        # buy 0.25 USD worth of the coin if the signal is 1
                        try:
                            if signal == 1:
                                # use order_crypto to buy the coin
                                order_crypto(
                                    symbol = coin,
                                    quantity_or_price = 0.25 * buying_limiter,
                                    amount_in = 'dollars',
                                    side = 'buy'
                                )
                            elif signal > 1:
                                # Buy 1% for every number over 1
                                # use order_crypto to buy the coin
                                order_crypto(
                                    symbol = coin,
                                    quantity_or_price = (0.01 * (signal - 1) + 0.01) * buying_power * buying_limiter,
                                    amount_in = 'dollars',
                                    side = 'buy'
                                )
                        except Exception as e:
                            print(f'Error: {e}')
                            logger.error(f"An error occurred: {e}, {traceback.format_exc()}, {dir(e)}")
                            continue

                    # We own the coin
                    # Case One: The price is below the lowest value in the last five hours, meaning the price is dropping and we should sell the coin if we bought it at a higher price, or buy the coin if we don't own it
                    list_prices = row['historical_prices']
                    list_five_latest = list_prices[-5:]
                    minimum = min(list_five_latest)
                    if current_price < minimum:
                        # Price is below the lowest value in the last five hours
                        if coin_holdings > 0:
                            # We own the coin
                            # Sell 100% of the holdings
                            # use order_crypto to sell the coin
                            order_crypto(
                                symbol = coin,
                                quantity_or_price = coin_holdings,
                                amount_in = 'quantity',
                                side = 'sell'
                            )
                            #log it
                            logger.info(f'Selling {coin_holdings} {coin} at {current_price}...')
                            # calculate the new buying power
                            buying_power += current_price * coin_holdings
                        else:
                            # We do not own the coin
                            # Calculate the amount of dollars to spend
                            dollars_spent = 0.005 * buying_power
                            print(f'Buying {dollars_spent} {coin} at {current_price}...')
                            # Buy the coin
                            # use order_crypto to buy the coin
                            order_crypto(
                                symbol=coin,
                                quantity_or_price=dollars_spent,
                                amount_in='dollars',
                                side='buy'
                            )
                            #log it
                            logger.info(f'Buying {dollars_spent} {coin} at {current_price}...')
                            buying_power -= dollars_spent
                    if signal == -1:
                        # Sell 70% of the holdings
                        # use order_crypto to sell the coin
                        order_crypto(
                            symbol=coin,
                            quantity_or_price=0.7 * coin_holdings,
                            amount_in='quantity',
                            side='sell'
                        )
                        # calculate the new buying power
                        buying_power += current_price * coin_holdings
                    if signal < -1:
                        # Sell 100% of the holdings
                        # use order_crypto to sell the coin
                        order_crypto(
                            symbol = coin,
                            quantity_or_price = coin_holdings,
                            amount_in = 'quantity',
                            side = 'sell'
                        )
                        #log it
                        logger.info(f'Selling {coin_holdings} {coin} at {current_price}...')
                        # calculate the new buying power
                        buying_power += current_price * coin_holdings
                    if signal == 1:
                        # Buy 50% of 1% of the buying power
                        # use r.orders to buy the coin
                        order_crypto(
                            symbol = coin,
                            quantity_or_price = 0.005 * buying_power,
                            amount_in = 'dollars',
                            side = 'buy'
                        )
                        #log it
                        logger.info(f'Buying {0.005 * buying_power} {coin} at {current_price}...')

                        print(f'Buying {0.005 * buying_power} {coin} at {current_price}...')
                        # reduce the buying power
                        buying_power -= 0.005 * buying_power
                    if signal > 1:
                        # Buy 1% for every number over 1
                        # use r.orders to buy the coin
                        order_crypto(
                            symbol = coin,
                            quantity_or_price = (0.01 * (signal - 1) + 0.01) * buying_power,
                            amount_in = 'dollars',
                            side = 'buy'
                        )
                        #log it
                        logger.info(f'Buying {(0.01 * (signal - 1) + 0.01) * buying_power} {coin} at {current_price}...')
                        print(f'Buying {(0.01 * (signal - 1) + 0.01) * buying_power} {coin} at {current_price}...')
                        ic()
                        # reduce the buying power
                        buying_power -= (0.01 * (signal - 1) + 0.01) * buying_power
                        # if r. orders .
                        # today_profit
                except Exception as e:
                    logger.error(f"An error occurred: {e}, {traceback.format_exc()}, {dir(e)}")
                    print(f'Error: {e}')
                    continue


        except Exception as e:
            logger.error(f"An error occurred: {e}, {traceback.format_exc()}, {dir(e)}")
                # Wait for ten minutes and show a loading bar
        # print out a detailed synopsis of the results so far
        print(f'Current holdings: {r.get_crypto_positions()}')
        print(f'Current buying power: {r.profiles.load_account_profile(info="crypto_buying_power")}')
        # profits
        # print(f'Current profits: {r.profiles.load_account_profile(info="crypto_day_traded_profit_loss")}')
        # wait for ten minutes
        # if daytime then wait 5 minutes
        # if night time then wait 30 minutes
        if is_daytime():
            print('Waiting 5 minutes...')
            for i in tqdm(range(60*5)):
                time.sleep(1)
        else:

            print('Waiting 30 minutes...')
            for i in tqdm(range(60*30)):
                time.sleep(1)


# %%
# emergency sell all crypto
if panic_mode:
    for index, row in crypto_data.iterrows():
        coin = row['ticker']
        coin_holdings = float(row['coin_holdings'])
        if coin_holdings > 0:
            r.order_sell_crypto_by_quantity(
                symbol=coin,
                quantity=coin_holdings,
                timeInForce='gtc',
                jsonify=True
            )
            print(f'Selling {coin_holdings} {coin}...')
        else:
            print(f'No holdings for {coin}...')


# %%
main_loop()

# %%
