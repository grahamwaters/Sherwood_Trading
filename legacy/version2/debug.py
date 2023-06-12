# %%
from robin_stocks import robinhood as r
import pandas as pd
from datetime import datetime
import logging
# import traceback # for debugging
from sys import exit # for debugging
import traceback # for debugging
import time
import ast
from os import path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from tqdm import tqdm
import numpy as np
from icecream import ic
from colorama import Fore, Back, Style
# Login to Robinhood
r.login('username', 'password')
panic_mode = False # set to True to sell all crypto positions immediately
# List of cryptocurrencies to trade
cryptos = ['BTC', 'ETH', 'ADA', 'DOGE', 'MATIC', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'SOL', 'AVAX', 'XLM', 'BCH', 'XTZ']
cryptos = ['BTC', 'ETH', 'ADA'] #todo remove this; it is for testing purposes only
crypto_data = pd.DataFrame()
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
buying_limiter = 0.60 # limits buys to 60% of buying power
# Initialize an empty dataframe to store crypto data
crypto_data = pd.DataFrame()
def get_crypto_data(cryptos, crypto_data):
    for crypto in tqdm(cryptos):
        try:
           # ic()
            #print(f'Getting data for {crypto}, please wait...',end='')
            # Get historical data
            historicals = r.get_crypto_historicals(crypto, interval='day', span='week')
            current_price = historicals[-1]['close_price']
            historical_prices = [x['close_price'] for x in historicals]
            # Load account profile
            profile = r.profiles.load_account_profile()
            buying_power = profile['buying_power']
           # ic()
            # Get crypto positions
            positions = r.crypto.get_crypto_positions()
            print(f'found {len(positions)} positions.')
            for position in tqdm(positions):
                # print(position)
                if position['currency']['code'] == crypto:
                    pos_dict = position['currency']
                    min_order_size = float(pos_dict['increment'])
                    coin_holdings = float(position['quantity_available'])
                   # ic()
           # ic()
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
           # ic()
            print(f'Error getting data for {crypto}: {e}')
            logger.error(f'Error getting data for {crypto}: {e}')
            traceback.print_exc()
            pass
    def add_me(crypto_data, metric, metric_name):
       # ic()
        if isinstance(metric, pd.Series):
           # ic()
            # if rsi is a series, get the last value
            crypto_data.loc[index, metric_name] = metric.iloc[-1]
        elif isinstance(metric, float):
           # ic()
            # if rsi is a float, just use it
            crypto_data.loc[index, metric_name] = metric
        elif isinstance(metric, np.ndarray):
           # ic()
            # if rsi is an array, get the last value
            crypto_data.loc[index, metric_name] = metric[-1]
        else:
            # otherwise, just use the value
            crypto_data.loc[index, metric_name] = metric
           # ic()
        return crypto_data
    # Calculate RSI, MACD, Bollinger Bands, etc. and add to dataframe
    for index, row in tqdm(crypto_data.iterrows()):
        prices = pd.Series(row['historical_prices'])
        # the ema below expects a series of floats, so convert if necessary
        if not isinstance(prices[0], float):
            prices = prices.apply(lambda x: float(x))
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
# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from colorama import Fore, Back
def calculate_pct_change_5h(historical_prices, current_price):
    try:
        historical_prices_last_5 = historical_prices[-5:]
        if all(price != 0.0 for price in historical_prices_last_5):
            return (current_price - historical_prices_last_5[-1]) / historical_prices_last_5[-1]
        else:
            for price in reversed(historical_prices_last_5):
                if price != 0.0:
                    return (current_price - price) / price
        return 0.0
    except Exception as e:
        print(e)
        logging.error(e)
        return 0.0
def convert_to_float_list(x):
    if isinstance(x, str):
        return [float(price) for price in x.strip('[]').split(',')]
    elif isinstance(x, list):
        return [float(i) for i in x]
    else:
        return [x]  # if x is not iterable, return it as a single item list
def update_signals(crypto_data):
    """
    The update_signals function takes in a dataframe of crypto_data and updates the signal column based on various conditions.
    The function first converts the current price to a float, then calculates take profit and stop loss values for each row.
    It then converts historical prices to floats, changes all columns except for symbol to numeric values,
    and calculates pct change 5h using calculate_pct_change_5h function. It sets all signals equal to 0 initially (no signal),
    then updates them based on different conditions such as rsi &lt; 30 &amp; macd &gt; 0 or current price &gt; ma200.
    :param crypto_data: Pass in the dataframe that contains all the crypto data
    :return: A dataframe with the following columns:
    :doc-author: Trelent
    """
    stop_loss = 0.95
    take_profit = 1.03
    crypto_data['current_price'] = [
       float(crypto_data['current_price'][0]) if isinstance(crypto_data['current_price'][0], str) else crypto_data['current_price'][0] for _ in range(len(crypto_data['current_price']))
    ]
    float(crypto_data['current_price'][0])
    crypto_data['take_profit_value'] = crypto_data['current_price'].apply(lambda x: max(1.00, take_profit*x))
    crypto_data['stop_loss_value'] = crypto_data['current_price'].apply(lambda x: stop_loss*x)
    crypto_data['historical_prices'] = crypto_data['historical_prices'].apply(convert_to_float_list)
    # ^ Convert all columns except for symbol to numeric values
    crypto_data[crypto_data.columns[1:]] = crypto_data[crypto_data.columns[1:]].apply(pd.to_numeric, errors='coerce')
    # ^ Calculate pct change 5h using calculate_pct_change_5h function
    crypto_data['pct_change_5h'] = crypto_data.apply(lambda x: calculate_pct_change_5h(x['historical_prices'], x['current_price']), axis=1)
    crypto_data['signal'] = 0 # 0 = no signal, 1 = buy, -1 = sell
    # Update signal based on different conditions
    # Simplified conditions checking and signal updates using vectorized operations
    crypto_data.loc[(crypto_data['rsi'] < 30) & (crypto_data['macd'] > 0), 'signal'] += 1
    crypto_data.loc[(crypto_data['rsi'] > 70) & (crypto_data['macd'] < 0), 'signal'] -= 1
    crypto_data.loc[(crypto_data['macd'] > crypto_data['macd_signal']) & (crypto_data['current_price'] > crypto_data['sma']), 'signal'] += 1
    crypto_data.loc[(crypto_data['macd'] < crypto_data['macd_signal']) & (crypto_data['current_price'] < crypto_data['sma']), 'signal'] -= 1
    crypto_data.loc[(crypto_data['current_price'] > crypto_data['ma200']), 'signal'] += 1
    crypto_data.loc[(crypto_data['current_price'] < crypto_data['ma200']), 'signal'] -= 1
    crypto_data.loc[(crypto_data['pct_change_5h'] > 0.05), 'signal'] += 1
    crypto_data.loc[(crypto_data['pct_change_5h'] < -0.05), 'signal'] -= 1
    # Stop loss and take profit conditions
    crypto_data['days_profit'] = (crypto_data['current_price'] / crypto_data['daily_return']) - 1
    crypto_data.loc[(crypto_data['daily_profit'] > crypto_data['take_profit_value']) | (crypto_data['days_profit'] > crypto_data['take_profit_value']), 'signal'] -= 1
    crypto_data.loc[(crypto_data['daily_profit'] < crypto_data['stop_loss_value']) | (crypto_data['days_profit'] < crypto_data['stop_loss_value']), 'signal'] -= 1
    # Save to csv
    crypto_data.to_csv('crypto_data.csv', index=False)
    return crypto_data
# %%
crypto_data = get_crypto_data(cryptos, crypto_data)
crypto_data.to_csv('crypto_data.csv', index=False)
# show the dataframe
crypto_data.head()
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
buying_limiter = 0.60 # only use 60% of the buying power
# Read the crypto_data.csv file with the correct data types
crypto_data = pd.read_csv('crypto_data.csv', converters={'historical_prices': eval})
buying_power = float(r.profiles.load_account_profile(info='crypto_buying_power')) * float(buying_limiter) # only use 60% of the buying power
def is_daytime():
    # CHECK TIME OF DAY IN CENTRAL STANDARD TIME
    current_time = datetime.datetime.now()
    if current_time.hour >= 11 and current_time.hour <= 23:
        return True
    else:
        return False
print(f'I can only play with {buying_power} dollars')
# %%
def main_loop():
    """
    The main_loop function is the main function that runs the entire program.
    It will run forever, and it will check for signals every 5 minutes during
    the day time (9am-4pm) and every 30 minutes at night. It will also cancel all orders before running.
    :return: The crypto_data dataframe
    :doc-author: Trelent
    """
    global crypto_data
    while True:
        # cancel all orders
        r.cancel_all_crypto_orders()
        time.sleep(10)
        # CHECK TIME OF DAY IN CENTRAL STANDARD TIME
        current_time = datetime.datetime.now()
        signal = 0 # initialize signal
        # IF ITS AFTER 11PM AND BEFORE 5AM, THEN THE MARKET IS CLOSED SO ONLY DEAL WITH CRYPTO, ALSO ALL SIGNALS SHOULD HAVE A REDUCED BUY_VOLUME BY REDUCING BUY_IN BY 90% KEEPING SELL VOLUMES THE SAME.
        buying_limiter = 1 if current_time.hour >= 23 or current_time.hour < 5 else 0.02
        crypto_data = get_crypto_data(cryptos, crypto_data)
        try:
            print('Updating signals...')
            try:
                update_signals(crypto_data)
            except Exception as e:
                logging.error(f'Error updating signals: {e}, {traceback.format_exc()}, {dir(e)}')
                print(f'Error updating signals: {e}, {traceback.format_exc()}, {dir(e)}')
                continue
            now = datetime.datetime.now()
            for index, row in tqdm(crypto_data.iterrows()):
                coin = row['ticker']
                signal = row['signal'] if row['signal'] != 'nan' else 0 #todo - this indicates an issue with crypto_df
                current_price = float(row['current_price'])
                buying_power = float(row['buying_power'])
                coin_holdings = float(row['coin_holdings'])
                crypto_positions = coin_holdings
                print(f' >> {row["ticker"]} << signal: {row["signal"]}, holding: {row["coin_holdings"]}, ${buying_power}')
                if coin_holdings > 0 and signal < 0:
                    #^ We own the coin and signal indicates selling
                    #^ sell 70% of our holdings, then update the buying power
                    quantity = 0.7 * coin_holdings if signal == -1 else coin_holdings
                    order_crypto(symbol=coin, quantity_or_price=quantity, amount_in='quantity', side='sell')
                    buying_power += current_price * quantity
                    logger.info(Fore.RED + f' [!] ---> Selling {quantity} {coin} at {current_price}...' + Fore.RESET)
                #^ if the signal is greater than 0, then buy the coin
                elif signal > 0:
                    # Signal indicates buying
                    amount = 0.005 * buying_power if signal == 1 else (0.01 * (signal - 1) + 0.01) * buying_power
                    order_crypto(symbol=coin, quantity_or_price=amount, amount_in='dollars', side='buy')
                    buying_power -= amount
                    logger.info(Fore.GREEN + f' [!] ---> Buying {amount} {coin} at {current_price}...' + Fore.RESET)
                # Handling lowest value in last 5 hours condition
                list_prices = row['historical_prices']
                #^ convert the string list_prices to a list
                list_prices = ast.literal_eval(list_prices)
                #^ get the last 5 prices
                list_five_latest = list_prices[-5:] #todo what units are these in?
                minimum = min(list_five_latest)
               # ic()
                if isinstance(minimum, str):
                   # ic()
                    minimum = float(minimum)
                    logging.info(f'Converting minimum: {minimum} to float... it was a string')
                else:
                   # ic()
                    try:
                        minimum = float(minimum)
                        logging.info(f'Converting minimum: {minimum} to float... it was a string')
                       # ic()
                    except Exception as e:
                        logging.info(f'Could not convert minimum: {minimum} to float... it was a string')
                        raise e #^ raise the error to the top level
                if current_price < minimum:
                   # ic()
                    if coin_holdings > 0:
                        order_crypto(symbol=coin, quantity_or_price=coin_holdings, amount_in='quantity', side='sell')
                        buying_power += current_price * coin_holdings
                        logger.info(Fore.RED + f' [!] ---> Selling {coin_holdings} {coin} at {current_price}...' + Fore.RESET)
                       # ic()
                    else:
                       # ic()
                        dollars_spent = 0.005 * buying_power
                        order_crypto(symbol=coin, quantity_or_price=dollars_spent, amount_in='dollars', side='buy')
                        buying_power -= dollars_spent
                        logger.info(Fore.GREEN + f' [!] ---> Buying {dollars_spent} {coin} at {current_price}...' + Fore.RESET)
                       # ic()
            tqdm.write(f'we have {buying_power} left')
        except Exception as e:
            logger.error(f"An error occurred: {e}, {traceback.format_exc()}, {dir(e)}")
        # Wait for ten minutes and show a loading bar
        #print(f'Current holdings: {r.get_crypto_positions()}')
        print(f'Current buying power: {r.profiles.load_account_profile(info="crypto_buying_power")}')
        print(f'Current profits: {r.profiles.load_account_profile(info="crypto_day_traded_profit_loss")}')
        # if daytime then wait 5 minutes
        # if night time then wait 30 minutes
        wait_time = 5 if is_daytime() else 30
        print(f'Waiting {wait_time} minutes...')
        for i in tqdm(range(60*wait_time)):
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
