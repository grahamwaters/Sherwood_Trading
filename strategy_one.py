#redo3.py
#Author: Graham Waters, 2023
#^ Essential Imports
import asyncio
import configparser
import logging
import os
import pytz
import random
import sys
import time
import traceback
from datetime import datetime
#^ Non-Essential Imports
import alive_progress
import pandas as pd
import pandas_ta
import pandas_ta as ta
import robin_stocks as rstocks
from alive_progress import alive_bar
from icecream import ic
from pytz import timezone
from robin_stocks import robinhood as r
from tqdm import tqdm
import numpy as np
#^ Constants
PCT_SPEND = 0.05 # 5% of buying power
verbose_mode = True # Set to True to see more logging output
from colorama import Back, Fore, Style
from ratelimit import limits, sleep_and_retry
## Load credentials from config/credentials.ini
config = configparser.ConfigParser()
config.read('config/credentials.ini')
# Define variables from config
coins = [coin.strip() for coin in config['trading']['coins'].split(', ')]
stop_loss_percent = float(config['trading']['stop_loss_percent'])
percent_to_use = float(config['trading']['percent_to_use'])
verbose_mode = config['logging'].getboolean('verbose_mode')
debug_verbose = config['logging'].getboolean('debug_verbose')
reset_positions = config['logging'].getboolean('reset_positions')
minimum_usd_per_position = float(config['trading']['minimum_usd_per_position'])
pct_to_buy_with = float(config['trading']['percent_to_use'])
# pct_to_buy_per_trade = float(config['trading']['percent_to_spend_per_trade'])
pct_to_buy_per_trade = 0.02
profit_threshold = 0.10
buy_in = 0.02 # 2% of the buying power that we have not already spent
# working_dataframe
working_dataframe = pd.DataFrame(columns=['coin', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'run_id', 'timestamp', 'price'])
# Log in to the Robinhood API
username = config['robinhood']['username']
password = config['robinhood']['password']
login = r.login(username, password)
print(f"Logged in as {username}")
# Function to get current price for a coin

total_money = float(r.load_account_profile()['crypto_buying_power']) * float(pct_to_buy_with)
# save the total money we have to spend to total_money.txt in data/ folder
with open('data/total_money.txt', 'w') as f:
    f.write(str(total_money))


@sleep_and_retry
def get_quantity(coin_name):
    positions = r.crypto.get_crypto_positions()
    positions_dict = {}
    for position in positions:
        positions_dict[position['currency']['code']] = float(position['quantity'])
    return positions_dict.get(coin_name, 0)

@sleep_and_retry
def get_current_price(coin_name):
    quote = r.crypto.get_crypto_quote(coin_name)
    return float(quote['mark_price'])
# Function to update the portfolio positions
@sleep_and_retry
def update_portfolio():
    positions = r.crypto.get_crypto_positions()
    positions_dict = {}
    for position in positions:
        positions_dict[position['currency']['code']] = float(position['quantity'])
    return positions_dict
# Function to buy a coin
@sleep_and_retry
def buy_coin(coin_name, amount, current_price):
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
    # save result to a file
    with open('data/buy_orders.txt', 'a') as f:
        f.write(f'{result}\n')
    return result
# Function to sell a coin
@sleep_and_retry
def sell_coin(coin_name):
    positions_dict = update_portfolio()
    amount_held = float(positions_dict.get(coin_name, 0))
    current_price = float(get_current_price(coin_name))
    result = r.orders.order_crypto(
        symbol=coin_name,
        amountIn='quantity',
        side='sell',
        quantityOrPrice=float(amount_held),
        limitPrice=float(current_price),
        timeInForce='gtc',
        jsonify=True
    )
    # save result to a file
    with open('data/sell_orders.txt', 'a') as f:
        f.write(f'{result}\n')
    return result
# Function to calculate technical indicators and generate trading signals
def calculate_ta_indicators(coins):
    global buy_in
    minimum_buy_usd = float(config['trading']['minimum_usd_per_trade'])
    # strip whitespace from the coins list
    coins = [coin.strip() for coin in coins]
    # load the working dataframe from the csv
    buy_in = float(buy_in) # 2% of the buying power that we have not already spent
    try:
        working_dataframe = pd.read_csv('data/working.csv')
    except:
        # create the working dataframe if it doesn't exist
        working_dataframe = pd.DataFrame(columns=['coin', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'run_id', 'timestamp', 'price'])
    pct_to_buy_with = float(config['trading']['percent_to_use']) if config['trading']['percent_to_use'] else 0.02
    with open('data/total_money.txt', 'r') as f:
        total_money = float(f.read())
    print(f'Total Money: ${total_money}')
    print(f'Available Buying Power: ${total_money}', end='')
    print(f' * {pct_to_buy_with} = ${total_money * pct_to_buy_with}')
    available_money = total_money * pct_to_buy_with
    print(f'Available Money: ${available_money}')
    print(f'Buy In: ${buy_in}')
    print(f'Available Money: ${available_money - buy_in}')

    for coin in coins:
        #* Stage One: Technical Analysis
        try:
            # Get the historical data for the coin
            historicals = r.get_crypto_historicals(coin, interval='5minute', span='week', bounds='24_7', info=None)
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(historicals)
            # Convert the 'begins_at' column to datetime and set it as the index
            df['begins_at'] = pd.to_datetime(df['begins_at'])
            df.set_index('begins_at', inplace=True)
            # Convert the data types of the columns to float
            df['open_price'] = df['open_price'].astype(float)
            df['close_price'] = df['close_price'].astype(float)
            df['high_price'] = df['high_price'].astype(float)
            df['low_price'] = df['low_price'].astype(float)
            # Calculate the RSI
            df['rsi'] = ta.momentum.rsi(df['close_price'], n=14, fillna=False)
            # Calculate the MACD
            macd_object = pandas_ta.macd(df['close_price'], fast=12, slow=26, signal=9, min_periods=None, fillna=False)
            df['macd_line'] = macd_object.MACD_12_26_9
            df['macd_signal'] = macd_object.MACDs_12_26_9
            df['macd_hist'] = macd_object.MACDh_12_26_9
            # Calculate the Bollinger Bands
            bbands_object = pandas_ta.bbands(df['close_price'], length=5, std=2.0, mamode='SMA', offset=0, append=True)
            df['BBM_5_2.0'] = bbands_object['BBM_5_2.0']
            df['BBU_5_2.0'] = bbands_object['BBU_5_2.0']
            df['BBL_5_2.0'] = bbands_object['BBL_5_2.0']
            # Calculate the EMA
            # an ema is a type of moving average that places a greater weight and significance on the most recent data points, making it a type of weighted average. We want to calculate the ema for the last 5 days
            df['ema'] = ta.ema(df['close_price'], length=5, fillna=False)
            #todo -- ALL_PATTERNS may give us access to things like the hammer pattern, which is a bullish indicator, 2crows... etc
            #* Stage Two: Generate Trading Signals
            sell_points = 0
            buy_points = 0
            #^ our buy_in will always be 2 % of the buying power that we have not already spent
            if buy_in < minimum_buy_usd:
                print(f'Buy In is less than the minimum buy amount of ${minimum_buy_usd}.')
                print(f'Increasing buy in to ${minimum_buy_usd}')
                buy_in = minimum_buy_usd
            # print(f'Checking Available Buying Power...')
            # cdl_pattern
            # bbands
            # When the price of the coin hits the upper bollinger band we want to trigger a sell signal
            if df['close_price'].iloc[-1] > df['BBU_5_2.0'].iloc[-1]:
                print(Fore.GREEN + f'Sell signal triggered for {coin}, price is above upper bollinger band\n\t purchased a lot of {coin} at {df["close_price"].iloc[-1]} totaling ${buy_in} on {datetime.now()}' + Style.RESET_ALL)
                # save the transaction to a log file as well called 'transactions.log'
                sell_coin(coin) #* Definite Sale
                # saving to the log
                with open('transactions.log', 'a') as f:
                    f.write(f'{coin} sold at {df["close_price"].iloc[-1]} on {datetime.now()}\n')
                sell_points -= 1 if sell_points > 0 else 0 # we don't want to go negative
            # When the price of the coin hits the lower bollinger band we want to trigger a buy signal
            if df['close_price'].iloc[-1] < df['BBL_5_2.0'].iloc[-1]:
                print(f'Buy signal triggered for {coin}, price is below lower bollinger band\n\t purchased a lot of {coin} at {df["close_price"].iloc[-1]} totaling ${buy_in} on {datetime.now()}')
                # buy_coin(coin, float(buy_in), df['close_price'].iloc[-1])
                buy_points += 1 #^ Potential Buy
                # available_money = float(available_money)
                # available_money -= buy_in
            # macd
            # quantity
            # when the macd line crosses the signal line from below we want to trigger a buy signal, and vice versa
            if df['macd_line'].iloc[-1] > df['macd_signal'].iloc[-1]:
                #print(f'Buy signal triggered for {coin}')
                # buy_coin(coin, float(buy_in), df['close_price'].iloc[-1])
                buy_points += 1 #^ Potential Buy
            if df['macd_line'].iloc[-1] < df['macd_signal'].iloc[-1]:
                # print(f'Sell signal triggered for {coin}')
                sell_points += 1 #^ Potential Sale
            # rsi
            if df['rsi'].iloc[-1] > 80:
                print(Fore.GREEN + f'Sell signal triggered for {coin}' + Fore.RESET)
                sell_coin(coin) #* Definite Sale
                # save to a log file
                with open('transactions.log', 'a') as f:
                    f.write(f'{coin} sold at {df["close_price"].iloc[-1]} on {datetime.now()}\n')
                sell_points -= 1 if sell_points > 0 else 0 # we don't want to go negative
            if df['rsi'].iloc[-1] < 20:
                print(f'Buy signal triggered for {coin}, price is below lower bollinger band\n\t purchased a lot of {coin} at {df["close_price"].iloc[-1]} totaling ${buy_in} on {datetime.now()}')
                buy_coin(coin, float(buy_in), df['close_price'].iloc[-1]) #* Definite Buy
                available_money -= buy_in
                # save to a log file
                with open('transactions.log', 'a') as f:
                    f.write(f'{coin} bought at {df["close_price"].iloc[-1]} on {datetime.now()}\n')
            # using levels from config
            if buy_points == 1:
                buy_in = float(config['levels']['1']) * available_money
            elif buy_points == 2:
                buy_in = float(config['levels']['2']) * available_money
            elif buy_points == 3:
                buy_in = float(config['levels']['3']) * available_money
            if buy_points > 0:
                if buy_in < minimum_buy_usd:
                    print(f'Buy In is less than the minimum buy amount of ${minimum_buy_usd}.')
                    print(f'Increasing buy in to ${minimum_buy_usd}')
                    buy_in = minimum_buy_usd
                print(Fore.GREEN + f'Buying {coin} because the buy points are greater than 0. Buy points: {buy_points}\n\t purchased a lot of {coin} at {df["close_price"].iloc[-1]} totaling ${buy_in} on {datetime.now()}' + Fore.RESET)
                buy_coin(coin, float(buy_in), df['close_price'].iloc[-1])
                # save a log file
                quantity = get_quantity(coin)
                with open('transactions.log', 'a') as f:
                    f.write(f'{coin} bought at {df["close_price"].iloc[-1]} on {datetime.now()}, quantity: {quantity}, purchased usd: {buy_in}, quantity: {quantity}, worth usd: ${df["close_price"].iloc[-1] * quantity}, buy_points: {buy_points}\n')
                available_money -= buy_in
                buy_points = 0 #& reset the buy points
            elif sell_points > 1:
                sell_coin(coin)
                print(Fore.RED + 'Selling {coin} because the sell points are greater than 1. Sell points: {sell_points}\n\t sold a lot of {coin} at {df["close_price"].iloc[-1]} totaling ${buy_in} on {datetime.now()}' + Fore.RESET)
                # save a log file
                quantity = get_quantity(coin)
                with open('transactions.log', 'a') as f:
                    f.write(f'{coin} sold at {df["close_price"].iloc[-1]} on {datetime.now()}, quantity: {quantity}, purchased usd: {buy_in}, sold usd: {df["close_price"].iloc[-1] * quantity}, profit: {df["close_price"].iloc[-1] * quantity - buy_in}, profit %: {((df["close_price"].iloc[-1] * quantity - buy_in) / buy_in) * 100}\n')
                sell_points = 0 #& reset the sell points
            else:
                print(f'No action taken for {coin}, buy points: {buy_points}, sell points: {sell_points}')
            # Append the new row to the working dataframe
            working_dataframe = working_dataframe.append(
                {
                    'coin': coin,
                    'price': df['close_price'].iloc[-1],
                    'rsi': df['rsi'].iloc[-1],
                    'macd_line': df['macd_line'].iloc[-1],
                    'macd_signal': df['macd_signal'].iloc[-1],
                    'macd_hist': df['macd_hist'].iloc[-1],
                    'BBM_5_2.0': df['BBM_5_2.0'].iloc[-1],
                    'BBU_5_2.0': df['BBU_5_2.0'].iloc[-1],
                    'BBL_5_2.0': df['BBL_5_2.0'].iloc[-1],
                    'ema': df['ema'].iloc[-1],
                    'timestamp': np.datetime64(datetime.now()),
                    'buy_points': buy_points,
                    'sell_points': sell_points
                },
                ignore_index=True
                )
            total_money = available_money
            # Save the working dataframe to a csv
            # print any signals that were triggered and what their names are
        except Exception as e:
            print(f'Error calculating technical indicators for {coin}: {e}')
            traceback.print_exc()  # This will print the traceback
            pass
        # except Exception as e:
        #     print(f'Error calculating technical indicators for {coin}: {e}')
        #     pass
    # save the results and data to working_dataframe
    working_dataframe.to_csv(f'data/working.csv', index=False)

    # print any signals that were triggered and what their names are
    print(f'Buy points: {buy_points}, Sell points: {sell_points}')
    for coin in coins:
        for signal in [
            'rsi', 'macd_line', 'macd_signal', 'macd_hist', 'BBM_5_2.0', 'BBU_5_2.0', 'BBL_5_2.0', 'ema'
            ]:
            if working_dataframe[working_dataframe['coin'] == coin][signal].iloc[-1] == 1:
                print(f'{coin} triggered a buy signal for {signal}')
            elif working_dataframe[working_dataframe['coin'] == coin][signal].iloc[-1] == -1:
                print(f'{coin} triggered a sell signal for {signal}')
    # print the total money
    account_money = float(r.load_account_profile()['crypto_buying_power'])
    print(f'Total money: ${total_money} [funds in reserve: ${account_money-total_money}]')
# Main loop to run the program indefinitely
from alive_progress import alive_bar, config_handler
# Set the bar style
config_handler.set_global(length=50, spinner='classic', bar='blocks')
while True:
    # read total_money from the `total_money.txt` file in the data folder
    with open('data/total_money.txt', 'r') as f:
        total_money = float(f.read())
    # cancel all outstanding orders


    # if the time is between 11:30 PM and 12:30 AM, then don't cancel orders, because our scheduled orders will be placed at 12:00 AM
    if pytz.timezone('US/Eastern').localize(datetime.now()).hour >= 23 or pytz.timezone('US/Eastern').localize(datetime.now()).hour < 1:
        print(f'Waiting for 30 minutes before canceling orders...')
        with alive_bar(60, title='Sleeping', bar='blocks', spinner='classic') as bar:
            for i in range(30):
                time.sleep(1)
        r.orders.cancel_all_crypto_orders()
    else:
        r.orders.cancel_all_crypto_orders()
        print(f'Canceled all orders')
    time.sleep(20)
    print(f'Calculating Indicators...')
    calculate_ta_indicators(coins)
    # check if it is daytime or nighttime (9:30 ET to 16:00 ET)
    if pytz.timezone('US/Eastern').localize(datetime.now()).hour >= 9 and pytz.timezone('US/Eastern').localize(datetime.now()).hour < 16: #& 9:30 ET to 16:00 ET
        daytime = True
        interval = int(config['trading']['daytime_interval'])
    else:
        daytime = False
        interval = int(config['trading']['nightime_interval'])
    print(f'Waiting for {interval} minutes...')
    interval = int(interval)
    with alive_bar(int(interval) * 60, title='Sleeping', bar='blocks', spinner='classic') as bar:
        for i in range(interval * 60):
            time.sleep(1)
            bar.text(f'Processing {i+1}/{int(interval) * 60}')
            bar()
    #^ Update any changes that we made to the config file
    config = configparser.ConfigParser()
    config.read('config/credentials.ini')
    # Define variables from config
    if pytz.timezone('US/Eastern').localize(datetime.now()).hour >= 9 and pytz.timezone('US/Eastern').localize(datetime.now()).hour < 16:
        percent_to_use = float(config['trading']['percent_to_use'])
        pct_to_buy_with = float(config['trading']['percent_to_use_night'])
        coins = [coin.strip() for coin in config['trading']['coins'].split(', ')]
        pct_to_buy_per_trade = float(config['trading']['percent_to_spend_per_trade'])
    else:
        percent_to_use = float(config['trading']['percent_to_use_night'])
        coins = config['trading']['night_time_coins']
        coins = [coin.strip() for coin in coins.split(', ')]
        pct_to_buy_per_trade = float(config['trading']['percent_to_spend_per_trade_night'])
        pct_to_buy_with = float(config['trading']['percent_to_use_night'])
    stop_loss_percent = float(config['trading']['stop_loss_percent'])

    verbose_mode = config['logging'].getboolean('verbose_mode')
    debug_verbose = config['logging'].getboolean('debug_verbose')
    reset_positions = config['logging'].getboolean('reset_positions')
    minimum_usd_per_position = float(config['trading']['minimum_usd_per_position'])

    if pytz.timezone('US/Eastern').localize(datetime.now()).hour >= 9 and pytz.timezone('US/Eastern').localize(datetime.now()).hour < 16:
        daytime = True
        interval = int(config['trading']['daytime_interval'])
        coins = [coin.strip() for coin in config['trading']['coins'].split(', ')]
        percent_to_spend_per_trade_day = float(config['trading']['percent_to_spend_per_trade'])
    else:
        daytime = False
        interval = int(config['trading']['nightime_interval'])
        coins = [coin.strip() for coin in config['trading']['nighttime_coins'].split(', ')]
        percent_to_spend_per_trade_night = float(config['trading']['percent_to_spend_per_trade_night'])



# Areas for future dev.
# trend
# volatility
# willr
# zscore
# volume
# atr
# ema
# candle_color
# ema
# high_low_range
# entropy
# max_drawdown
# min_drawdown
# momentum