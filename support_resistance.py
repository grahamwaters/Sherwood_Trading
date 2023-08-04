import asyncio
from robin_stocks import robinhood as r
import numpy as np
import pandas as pd
import configparser
import logging
import time
import sys
import os
import datetime
from datetime import datetime
from datetime import timedelta
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
from colorama import Fore, Back, Style
from legacy.V1.supportmaterial import d
config = configparser.ConfigParser()
config.read('config/credentials.ini')

coins = [coin.strip() for coin in config['trading']['coins'].split(', ')]
stop_loss_percent = float(config['trading']['stop_loss_percent'])
percent_to_use = float(config['trading']['percent_to_use'])
verbose_mode = config['logging'].getboolean('verbose_mode')
debug_verbose = config['logging'].getboolean('debug_verbose')
reset_positions = config['logging'].getboolean('reset_positions')
minimum_usd_per_position = float(config['trading']['minimum_usd_per_position'])
minimum_usd_per_trade = float(config['trading']['minimum_usd_per_trade'])
pct_to_buy_with = float(config['trading']['percent_to_use'])
pct_to_buy_per_trade = float(config['trading']['percent_to_spend_per_trade'])
username = config['robinhood']['username']
password = config['robinhood']['password']
login = r.login(username, password)

# Set up logging
logging.basicConfig(filename='crypto_trading.log', level=logging.INFO)
logging.info(f"Logged in as {username}")

print(f'coins: {coins}')
print(f'stop_loss_percent: {stop_loss_percent}')
print(f'=' * 20)

@sleep_and_retry
@limits(calls=5, period=1)
def get_current_price(symbol):
    current_price = r.crypto.get_crypto_quote(symbol, info='mark_price')
    return float(current_price)

# Get crypto positions
positions = r.crypto.get_crypto_positions()
print(f'found {len(positions)} positions.')

# Initialize an empty dictionary to store the minimum order sizes
min_order_sizes = {}

for position in positions:
    # Extract the cryptocurrency code and minimum order size
    crypto_code = position['currency']['code']
    min_order_size = float(position['currency']['increment'])

    # Add the minimum order size to the dictionary
    min_order_sizes[crypto_code] = min_order_size

print(min_order_sizes)
time.sleep(1)

# Function to buy a coin
@sleep_and_retry
@limits(calls=5, period=1)
def buy_coin(coin_name, amount, minimum_usd_per_trade=minimum_usd_per_trade, verbose_mode=False):

    global min_order_sizes

    # if the amount is more precise than the minimum order size, round it to the nearest order sizes increment
    if amount < min_order_sizes[coin_name]:
        # set the amount to the minimum order size (in dollars)
        amount = round(min_order_sizes[coin_name],2)
    else:
        # round the amount to the nearest order size increment (e.g. $0.01 for BTC)
        amount = round((amount / min_order_sizes[coin_name]) * min_order_sizes[coin_name],2)
    buying_usd = max(float(amount), minimum_usd_per_trade)
    if verbose_mode: print(Fore.BLUE + f'buying_usd: {buying_usd} | minimum_usd_per_trade: {minimum_usd_per_trade}' + Fore.RESET)
    try:
        current_price = get_current_price(coin_name)
        # I am buying X dollars (usd) of the coin at the current price which makes a batch of N coins.
        N_Coins = round(float(buying_usd) / float(current_price), 8) # this is the amount of coins I am buying
        X_dollars = round(float(N_Coins) * float(current_price), 2) # this is the actual amount of dollars I am spending
        result = r.orders.order_buy_crypto_by_price(
            symbol=str(coin_name).upper(),
            amountInDollars=float(X_dollars),
            timeInForce='gtc',
            jsonify=True
        )
        if len(result) > 0:
            # save result to a file
            with open('data/buy_orders.txt', 'a') as f:
                f.write(f'{result}\n')
            with open('data/transactions.csv','a') as c:
                c.write(f'{datetime.now()},{coin_name},{current_price},{N_Coins},{X_dollars},"buy",{buying_usd}\n')
            if verbose_mode:
                print(Fore.GREEN + f' > bought {buying_usd} {coin_name} for ${X_dollars} at {datetime.now()}' + Fore.RESET)
        else:
            if verbose_mode:
                print(Fore.RED + f' > error buying {coin_name} at {datetime.now()}' + Fore.RESET)
                print(result)
        return result
    except Exception as e:
        logging.error(e)
        error = f'Error buying {coin_name} at {datetime.now()}'
        # save error to a file
        with open('data/errors.txt', 'a') as f:
            f.write(f'{error}\n')
        return error
# Function to sell a coin
@sleep_and_retry
@limits(calls=5, period=1)
def sell_coin(coin_name, stop_loss_trigger_price=None, verbose_mode=False):
    #* SCENARIO B: A Stop Loss Order
    if stop_loss_trigger_price is not None:
        try:
            holdings = r.crypto.get_crypto_positions(info=None)
            coin_holdings = [coin for coin in holdings if coin['currency']['code'] == coin_name]
            amount_to_sell = float(coin_holdings[0]['quantity'])
            if verbose_mode:
                print(Fore.YELLOW + f' > we have {amount_to_sell} {coin_name} to sell' + Fore.RESET)
            result = r.orders.order_sell_crypto_by_price(
                symbol=coin_name,
                amountInDollars=float(amount_to_sell),
                priceType='bid_price',
                timeInForce='gtc',
                jsonify=True
            )
            return result
        except Exception as e:
            logging.error(e)
            error = f'Error selling {coin_name} at {datetime.now()}'
            # save error to a file
            with open('data/errors.txt', 'a') as f:
                f.write(f'{error}\n')
            return error
    #* SCENARIO A: A normal Sell Order
    try:
        current_price = get_current_price(coin_name)
        holdings = r.crypto.get_crypto_positions(info=None)
        coin_holdings = [coin for coin in holdings if coin['currency']['code'] == coin_name]
        amount_to_sell = float(coin_holdings[0]['quantity'])
        result = r.orders.order_sell_crypto_by_price(
            symbol=coin_name,
            amountInDollars=float(amount_to_sell),
            priceType='bid_price',
            timeInForce='gtc',
            jsonify=True
        )
        return result
    except Exception as e:
        logging.error(e)
        error = f'Error selling {coin_name} at {datetime.now()}'
        # save error to a file
        with open('data/errors.txt', 'a') as f:
            f.write(f'{error}\n')
        return error

async def place_order(symbol, order_type, price, stop_loss=None, take_profit=None):
    if order_type == 'buy':
        #^note that buys must be above at least the minimum_usd_per_trade OR $1.00 whichever is greater
        amountInDollars= max(float(minimum_usd_per_trade), 1.00)
        buy_coin(coin_name = symbol, amount = amountInDollars)
        logging.info(f"Placed buy order for {symbol} at price {price}, cost ${price}")
        print(f"Placed buy order for {symbol} at price {price}, cost ${price}")
        if stop_loss is not None:
            # this means we have a stop loss % being passed (in the format 0.80 for 80%) which needs to be calculated >> and then a sell order placed at that price
            stop_loss = float(stop_loss)
            stop_loss_trigger_price = round(price * float(stop_loss),2)
            # place a sell order for the coin at the stop loss trigger price
            sell_coin(coin_name = symbol,
                        stop_loss_trigger_price = stop_loss_trigger_price)  # this is the price at which the sell order will be triggered
            print(Fore.YELLOW + f"Placed stop loss order for {symbol} at price {stop_loss_trigger_price}, cost ${stop_loss_trigger_price}" + Fore.RESET)
    elif order_type == 'sell':
        quantity = float(r.crypto.get_crypto_positions(symbol)[0]['quantity'])
        sell_coin(coin_name = symbol)
        print(f"Placed sell order for {symbol} at quantity {quantity}, cost ${quantity*price}")

async def place_orders(symbol, support, resistance, threshold):
    current_price = get_current_price(symbol)
    if abs(current_price - support) <= threshold:
        await place_order(symbol, 'buy', current_price)
    elif abs(current_price - resistance) <= threshold:
        await place_order(symbol, 'sell', current_price)
    elif current_price > resistance:
        await place_order(symbol, 'buy', current_price)
    elif current_price < support:
        await place_order(symbol, 'sell', current_price)

def calculate_rsi(data, period):
    """
    The calculate_rsi function takes in a dataframe and a period, and returns the RSI for that period.
    The function first calculates the difference between each row of data, then creates two new columns: up and down.
    Up is equal to all positive values from delta, while down is equal to all negative values from delta.
    Then we calculate average_gain by taking the mean of up over our specified window (period). We do this again with average_loss but take absolute value so we don't have any negative numbers.

    :param data: Pass in the dataframe that contains the coin price
    :param period: Determine the number of days used to calculate the rsi
    :return: A series with the rsi values for each day
    :doc-author: Trelent
    """

    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window=period).mean()
    average_loss = abs(down.rolling(window=period).mean())
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(dataframe_list, short_period, long_period):
    """
    The calculate_macd function takes in a dataframe and two periods, short_period and long_period.
    It then calculates the exponential moving average of the dataframe using both periods.
    The difference between these two averages is called the macd line. The signal line is calculated by taking an EMA of this macd line.

    :param data: Pass in the dataframe, short_period is used to set the period for the short ema and long_period is used to set the period for long ema
    :param short_period: Calculate the short-term exponential moving average (ema)
    :param long_period: Calculate the long exponential moving average
    :return: A tuple of two pandas series
    :doc-author: Trelent
    """
    # the
    return macd_line, signal_line

def calculate_ema(data, window):
    """
    The calculate_ema function takes in a dataframe and a window size, and returns the exponential moving average of that dataframe.

    :param data: Pass in the dataframe that we want to calculate the ema for
    :param window: Specify the number of days to use in the calculation
    :return: The exponential moving average of the data
    :doc-author: Trelent
    """

    return data.ewm(span=window, adjust=False).mean()

def get_our_data(coin):
    # Download historical data
    data = r.crypto.get_crypto_historicals(str(coin).upper(), interval='5minute', span='week', info=None)
    # Calculate MACD
    # convert 'close_price','open_price','high_price','low_price' to float
    # go through each of the rows in the columns and convert to float
    for row in data:
        for key in row:
            try:
                row[key] = float(row[key])
            except:
                pass

    dfs = [pd.DataFrame(data) for df in data]
    data = pd.concat(dfs)

    # save taht data to a csv file
    data.to_csv(f'data/coins/{str(coin).lower()}.csv')

    # calculate the macd and signal lines
    # we have data in the format of a dataframe
    # data['close_price'].iloc[-1] for example accesses the last close price in the dataframe. We can use this to calculate the macd and signal lines
    # we need to calculate the ema for the short and long periods
    ema_short_period = 12
    ema_long_period = 26
    # without using the function
    ema_short = data['close_price'].ewm(span=ema_short_period, adjust=False).mean()
    ema_long = data['close_price'].ewm(span=ema_long_period, adjust=False).mean()
    # now macd without using the function
    macd_line = ema_short.iloc[-1] - ema_long.iloc[-1]
    signal_line = macd_line.ewm(span=9, adjust=False).mean()


async def main(symbol, interval='5minute'):
    historical_data = r.crypto.get_crypto_historicals(
        symbol = symbol
        , interval=interval, span='day')
    closing_prices = [float(x['close_price']) for x in historical_data]
    support = min(closing_prices)
    resistance = max(closing_prices)
    await place_orders(symbol, support, resistance, threshold=0.01)

# Log initial portfolio
portfolio = r.account.build_holdings()
logging.info(f"Initial portfolio: {portfolio}")

# Log initial cash
cash = r.profiles.load_account_profile(info='portfolio_cash')
cash = float(cash)
logging.info(f"Initial cash: {cash}")

while True:

    # update the ini variables
    coins = [coin.strip() for coin in config['trading']['coins'].split(', ')]
    # stop_loss_percent = float(config['trading']['stop_loss_percent'])
    # percent_to_use = float(config['trading']['percent_to_use'])
    # verbose_mode = config['logging'].getboolean('verbose_mode')
    # debug_verbose = config['logging'].getboolean('debug_verbose')
    # reset_positions = config['logging'].getboolean('reset_positions')
    # minimum_usd_per_position = float(config['trading']['minimum_usd_per_position'])
    # minimum_usd_per_trade = float(config['trading']['minimum_usd_per_trade'])
    # pct_to_buy_with = float(config['trading']['percent_to_use'])
    # pct_to_buy_per_trade = float(config['trading']['percent_to_spend_per_trade'])
    # import all variables from the config file
    print(f'Importing trading variables from config file at {datetime.now()}')
    for key in tqdm(config['trading']):
        globals()[key] = config['trading'][key]
    print(f'Importing logging variables from config file at {datetime.now()}')
    for key in config['logging']:
        globals()[key] = config['logging'][key]

    # fix the coins list
    coins = [coin.strip() for coin in coins.split(', ')]


    print(f'Running at {datetime.now()}')
    for coin in tqdm(coins):
        print(f'Running for {coin}')
        asyncio.run(main(coin))

    # pause trading for the interval specified in the config file
    pause_interval = float(config['trading']['daytime_interval'])
    time.sleep(60*pause_interval)  # Run every ten minutes

    # Log portfolio and cash after each round of trading
    portfolio = r.account.build_holdings()
    logging.info(f"Current portfolio: {portfolio}")
    cash = r.profiles.load_account_profile(info='portfolio_cash')
    logging.info(f"Current cash: {cash}")
