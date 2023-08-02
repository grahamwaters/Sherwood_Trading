import json
import logging
import pandas as pd
import pandas_ta as ta
import robin_stocks as rstocks
from robin_stocks import robinhood as r
from datetime import datetime
from pytz import timezone
import asyncio
import uuid
import time
import math
import os
from tqdm import tqdm
from colorama import Fore, Back, Style
from icecream import ic
import configparser
import sys
import traceback
import warnings
import alive_progress
from alive_progress import alive_bar
import random
import numpy as np
from time import sleep
from datetime import datetime

logging.basicConfig(level=logging.INFO)
@sleep_and_retry
def buy_coin(coin_name, amount):
    try:
        current_price = get_current_price(coin_name)
        amount_to_buy = float(amount) / float(current_price)
        buying_usd = round(float(amount_to_buy), 2)
        buy_cost = round(float(buying_usd) * float(current_price), 2)
        result = r.orders.order_crypto(
            symbol=str(coin_name).upper(),
            amountIn='dollars',
            side='buy',
            quantityOrPrice=float(buy_cost),
            limitPrice=float(current_price),
            timeInForce='gtc',
            jsonify=True
        )
        return result
    except Exception as e:
        print(f"An error occurred when trying to buy {coin_name}: {e}")
        return None

@sleep_and_retry
def sell_coin(coin_name):
    try:
        current_price = get_current_price(coin_name)
        holdings = r.crypto.get_crypto_positions(info=None)
        coin_holdings = [coin for coin in holdings if coin['currency']['code'] == coin_name]
        amount_to_sell = float(coin_holdings[0]['quantity'])
        result = r.orders.order_crypto(
            symbol=coin_name,
            amountIn='quantity',
            side='sell',
            quantityOrPrice=float(amount_to_sell),
            limitPrice=float(current_price),
            timeInForce='gtc',
            jsonify=True
        )
        return result
    except Exception as e:
        print(f"An error occurred when trying to sell {coin_name}: {e}")
        return None

import pandas as pd
import numpy as np

def calculate_rsi(data, period):
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window=period).mean()
    average_loss = abs(down.rolling(window=period).mean())
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_period, long_period):
    short_ema = data.ewm(span=short_period, adjust=False).mean()
    long_ema = data.ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

import yfinance as yf

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Download historical data
data = yf.download('BTC-USD', start='2020-01-01', end='2022-12-31')

# Calculate MACD
macd_line, signal_line = calculate_macd(data['Close'], 12, 26)
import pandas as pd
import yfinance as yf
from pyti.relative_strength_index import relative_strength_index as rsi

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def calculate_macd(data, short_window, long_window):
    short_ema = calculate_ema(data, short_window)
    long_ema = calculate_ema(data, long_window)
    macd_line = short_ema - long_ema
    signal_line = calculate_ema(macd_line, 9)
    return macd_line, signal_line

# Download historical data
data = yf.download('BTC-USD', start='2020-01-01', end='2022-12-31')

# Calculate MACD
macd_line, signal_line = calculate_macd(data['Close'], 12, 26)

# Calculate RSI
data['RSI'] = rsi(data['Close'], 14)

# Initialize holding status
holding = False

# Iterate over data
for i in range(1, len(data)):
    # Check if RSI is below 20
    if data['RSI'].iloc[i] < 20:
        # Check if MACD line crosses signal line from below
        if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] < signal_line.iloc[i-1]:
            if holding:
                print(f"Sell at {data['Close'].iloc[i]}")
                holding = False
        # Check if MACD line crosses signal line from above
        elif macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] > signal_line.iloc[i-1]:
            if holding:
                print(f"Sell at {data['Close'].iloc[i]}")
                holding = False

from scipy.signal import argrelextrema

def get_local_peaks(data):
    peaks = argrelextrema(data, np.greater)
    return peaks
def calculate_percentage_change(old_price, new_price):
    return ((new_price - old_price) / old_price) * 100

from datetime import datetime, timedelta
import pytz

# Function to sell a coin
@sleep_and_retry
def sell_coin(coin_name):
    current_price = float(get_current_price(coin_name))
    holdings = r.crypto.get_crypto_positions(info=None)
    coin_holdings = [coin for coin in holdings if coin['currency']['code'] == coin_name]
    amount_to_sell = float(coin_holdings[0]['quantity'])
    result = r.orders.order_crypto(
        symbol=coin_name,
        amountIn='quantity',
        side='sell',
        quantityOrPrice=float(amount_to_sell),
        limitPrice=float(current_price),
        timeInForce='gtc',
        jsonify=True
    )
    return result

# Function to implement the stop loss strategy
def stop_loss_strategy(coin_name, percentage_drop_threshold=5):
    # Get the current time in EST
    current_time = datetime.now(pytz.timezone('US/Eastern'))

    # Check if it's 11 PM EST
    if current_time.hour == 23:
        # Get the current price and the price 24 hours ago
        current_price = float(get_current_price(coin_name))
        past_price = float(get_past_price(coin_name, current_time - timedelta(days=1)))

        # Calculate the percentage change
        percentage_change = calculate_percentage_change(past_price, current_price)

        # If the price has dropped by more than the threshold, sell the coin
        if percentage_change < -percentage_drop_threshold:
            sell_coin(coin_name)

import pandas as pd
import yfinance as yf
from pyti.relative_strength_index import relative_strength_index as rsi


from scipy.signal import argrelextrema

def get_local_peaks(data):
    peaks = argrelextrema(data, np.greater)
    return peaks

def calculate_percentage_change(old_price, new_price):
    return ((new_price - old_price) / old_price) * 100

from datetime import datetime, timedelta
import pytz

# Function to sell a coin
@sleep_and_retry
def sell_coin(coin_name):
    current_price = float(get_current_price(coin_name))
    holdings = r.crypto.get_crypto_positions(info=None)
    coin_holdings = [coin for coin in holdings if coin['currency']['code'] == coin_name]
    amount_to_sell = float(coin_holdings[0]['quantity'])
    result = r.orders.order_crypto(
        symbol=coin_name,
        amountIn='quantity',
        side='sell',
        quantityOrPrice=float(amount_to_sell),
        limitPrice=float(current_price),
        timeInForce='gtc',
        jsonify=True
    )
    return result

# Function to implement the stop loss strategy
def stop_loss_strategy(coin_name, percentage_drop_threshold=5):
    # Get the current time in EST
    current_time = datetime.now(pytz.timezone('US/Eastern'))

    # Check if it's 11 PM EST
    if current_time.hour == 23:
        # Get the current price and the price 24 hours ago
        current_price = float(get_current_price(coin_name))
        past_price = float(get_past_price(coin_name, current_time - timedelta(days=1)))

        # Calculate the percentage change
        percentage_change = calculate_percentage_change(past_price, current_price)

        # If the price has dropped by more than the threshold, sell the coin
        if percentage_change < -percentage_drop_threshold:
            sell_coin(coin_name)
def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def calculate_macd(data, short_window, long_window):
    short_ema = calculate_ema(data, short_window)
    long_ema = calculate_ema(data, long_window)
    macd_line = short_ema - long_ema
    signal_line = calculate_ema(macd_line, 9)
    return macd_line, signal_line

# Download historical data
data = yf.download('BTC-USD', start='2020-01-01', end='2022-12-31')

# Calculate MACD
macd_line, signal_line = calculate_macd(data['Close'], 12, 26)

# Calculate RSI
data['RSI'] = rsi(data['Close'], 14)

# Initialize holding status
holding = False

# Iterate over data
for i in range(1, len(data)):
    # Check if RSI is below 20
    if data['RSI'].iloc[i] < 20:
        # Check if MACD line crosses signal line from below
        if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] < signal_line.iloc[i-1]:
            if holding:
                print(f"Sell at {data['Close'].iloc[i]}")
                holding = False
        # Check if MACD line crosses signal line from above
        elif macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] > signal_line.iloc[i-1]:
            if holding:
                print(f"Sell at {data['Close'].iloc[i]}")
                holding = False



def strategy_1(coin):
    logging.info(f"Running strategy 1 for {coin}")

    # Get the RSI data
    price_data = r.crypto.get_crypto_historicals(coin, interval='hour', span='day')
    price_data = pd.DataFrame(price_data)
    price_data['close_price'] = price_data['close_price'].astype(float)
    rsi_data = calculate_rsi(price_data['close_price'], 14)

    # Check if RSI goes below 20
    if rsi_data[-1] < 20:
        logging.info(f"RSI for {coin} is below 20")

        # Watch for a RSI local peak that is above the lowest RSI
        rsi_local_high = get_local_peaks(rsi_data)

        # Monitor at five minute intervals until the RSI dips below the mean of the last five RSI values
        while True:
            rsi_data = calculate_rsi(price_data['close_price'], 14)
            if rsi_data[-1] < np.mean(rsi_data[-5:]):
                logging.info(f"RSI for {coin} has dipped below the mean of the last five values")

                # Trigger a market sell of the coin
                sell_coin(coin)
                break

        # If the difference between the first and last of the last three RSI measurements is greater than the standard distance between the last five
        if abs(rsi_data[-3] - rsi_data[-1]) > np.std(rsi_data[-5:]):
            logging.info(f"The difference between the first and last of the last three RSI measurements for {coin} is greater than the standard distance between the last five")

            # Trigger an immediate market sell of the coin
            sell_coin(coin)

# Function to implement Strategy 2
def strategy_2(coin):
    # Get the RSI data
    rsi_data = calculate_rsi(coin, 14)

    # Check if RSI goes above 80
    if rsi_data[-1] > 80:
        # Trigger a limit sell of the coin
        sell_coin(coin)
import robin_stocks as r

def trailing_stop_loss(coin_name, percentage_drop):
    holdings = r.crypto.get_crypto_positions(info=None)
    coin_holdings = [coin for coin in holdings if coin['currency']['code'] == coin_name]
    if not coin_holdings:
        print(f"No holdings for {coin_name}")
        return
    initial_price = float(coin_holdings[0]['cost_bases'][0]['direct_cost_basis'])
    P_lowest = initial_price
    while True:
        current_price = get_current_price(coin_name)
        if current_price < P_lowest:
            P_lowest = current_price
        if current_price < P_lowest * (1 - percentage_drop/100):
            result = sell_coin(coin_name)
            print(f"Sold {coin_name} at {current_price} due to trailing stop loss.")
            return result
        time.sleep(300)  # sleep for 5 minutes

def get_current_price(coin_name):
    return float(r.crypto.get_crypto_quote(coin_name, info='mark_price'))

def sell_coin(coin_name):
    current_price = get_current_price(coin_name)
    holdings = r.crypto.get_crypto_positions(info=None)
    coin_holdings = [coin for coin in holdings if coin['currency']['code'] == coin_name]
    amount_to_sell = float(coin_holdings[0]['quantity'])
    result = r.orders.order_crypto(
        symbol=coin_name,
        amountIn='quantity',
        side='sell',
        quantityOrPrice=float(amount_to_sell),
        limitPrice=float(current_price),
        timeInForce='gtc',
        jsonify=True
    )
    return result
import time
import pytz
from datetime import datetime
from talib import RSI
from robin_stocks.robinhood import crypto

# Define the RSI limits
rsi_low_limit = 20
rsi_high_limit = 70

# Define the RSI strength threshold
rsi_strength_threshold = 2.0

# Function to calculate RSI Strength
def calculate_rsi_strength(coin_name):
    rsi_strength = 0.0
    rsi_prev = 0.0

    # Get the historical prices for the coin in the last 2 hours
    historical_prices = crypto.get_crypto_historicals(coin_name, interval='5minute', span='2hour')

    # Calculate the RSI for each price point
    close_prices = [float(data['close_price']) for data in historical_prices]
    rsi_values = RSI(close_prices)

    # Calculate the RSI strength
    for rsi in rsi_values:
        if rsi < rsi_low_limit and rsi_prev > rsi_high_limit:
            rsi_strength += 1.0
        rsi_prev = rsi

    return rsi_strength

# Function to sell a coin
def sell_coin(coin_name):
    current_price = float(crypto.get_crypto_quote(coin_name)['mark_price'])
    holdings = crypto.get_crypto_positions(info=None)
    coin_holdings = [coin for coin in holdings if coin['currency']['code'] == coin_name]
    amount_to_sell = float(coin_holdings[0]['quantity'])
    result = crypto.order_sell_crypto_limit(coin_name, amount_to_sell, current_price)
    return result

# Main function to implement the strategy
def rsi_strength_strategy(coin_name):
    while True:
        rsi_strength = calculate_rsi_strength(coin_name)
        if rsi_strength > rsi_strength_threshold:
            sell_result = sell_coin(coin_name)
            print(f'Sold {coin_name} due to high RSI strength: {rsi_strength}')
            print(sell_result)
        time.sleep(300)  # Sleep for 5 minutes
