from finta import TA
from colorama import Fore, Back, Style
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import traceback
from robin_stocks import robinhood as r
import json
import time
# import timezone
import pytz
from pytz import timezone
from datetime import datetime

import csv
from time import sleep
import os
from ratelimit import limits, sleep_and_retry
signals_dict = {} # initialize the signals dictionary
minimum_orders_coins= {}
from datetime import datetime
from tqdm import tqdm



"""
finta supports the following indicators:
* Simple Moving Average 'SMA'
* Simple Moving Median 'SMM'
* Smoothed Simple Moving Average 'SSMA'
* Exponential Moving Average 'EMA'
* Double Exponential Moving Average 'DEMA'
* Triple Exponential Moving Average 'TEMA'
* Triangular Moving Average 'TRIMA'
* Triple Exponential Moving Average Oscillator 'TRIX'
* Volume Adjusted Moving Average 'VAMA'
* Kaufman Efficiency Indicator 'ER'
* Kaufman's Adaptive Moving Average 'KAMA'
* Zero Lag Exponential Moving Average 'ZLEMA'
* Weighted Moving Average 'WMA'
* Hull Moving Average 'HMA'
* Elastic Volume Moving Average 'EVWMA'
* Volume Weighted Average Price 'VWAP'
* Smoothed Moving Average 'SMMA'
* Fractal Adaptive Moving Average 'FRAMA'
* Moving Average Convergence Divergence 'MACD'
* Percentage Price Oscillator 'PPO'
* Volume-Weighted MACD 'VW_MACD'
* Elastic-Volume weighted MACD 'EV_MACD'
* Market Momentum 'MOM'
* Rate-of-Change 'ROC'
* Relative Strenght Index 'RSI'
* Inverse Fisher Transform RSI 'IFT_RSI'
* True Range 'TR'
* Average True Range 'ATR'
* Stop-and-Reverse 'SAR'
* Bollinger Bands 'BBANDS'
* Bollinger Bands Width 'BBWIDTH'
* Momentum Breakout Bands 'MOBO'
* Percent B 'PERCENT_B'
* Keltner Channels 'KC'
* Donchian Channel 'DO'
* Directional Movement Indicator 'DMI'
* Average Directional Index 'ADX'
* Pivot Points 'PIVOT'
* Fibonacci Pivot Points 'PIVOT_FIB'
* Stochastic Oscillator %K 'STOCH'
* Stochastic oscillator %D 'STOCHD'
* Stochastic RSI 'STOCHRSI'
* Williams %R 'WILLIAMS'
* Ultimate Oscillator 'UO'
* Awesome Oscillator 'AO'
* Mass Index 'MI'
* Vortex Indicator 'VORTEX'
* Know Sure Thing 'KST'
* True Strength Index 'TSI'
* Typical Price 'TP'
* Accumulation-Distribution Line 'ADL'
* Chaikin Oscillator 'CHAIKIN'
* Money Flow Index 'MFI'
* On Balance Volume 'OBV'
* Weighter OBV 'WOBV'
* Volume Zone Oscillator 'VZO'
* Price Zone Oscillator 'PZO'
* Elder's Force Index 'EFI'
* Cummulative Force Index 'CFI'
* Bull power and Bear Power 'EBBP'
* Ease of Movement 'EMV'
* Commodity Channel Index 'CCI'
* Coppock Curve 'COPP'
* Buy and Sell Pressure 'BASP'
* Normalized BASP 'BASPN'
* Chande Momentum Oscillator 'CMO'
* Chandelier Exit 'CHANDELIER'
* Qstick 'QSTICK'
* Twiggs Money Index 'TMF'
* Wave Trend Oscillator 'WTO'
* Fisher Transform 'FISH'
* Ichimoku Cloud 'ICHIMOKU'
* Adaptive Price Zone 'APZ'
* Squeeze Momentum Indicator 'SQZMI'
* Volume Price Trend 'VPT'
* Finite Volume Element 'FVE'
* Volume Flow Indicator 'VFI'
* Moving Standard deviation 'MSD'
* Schaff Trend Cycle 'STC'
* Mark Whistler's WAVE PM 'WAVEPM'
"""
### ACTION FUNCTIONS ###
@sleep_and_retry
def order_crypto(symbol, quantity_or_price, amount_in='dollars', side='buy', bp=None, timeInForce='gtc'):
    """
    The order_crypto function is used to place a crypto order.

    :param symbol: Specify the coin you want to buy
    :param quantity_or_price: Specify the amount of crypto you want to buy or sell
    :param amount_in: Specify the amount of money you want to spend on buying a coin
    :param side: Specify whether you want to buy or sell the coin
    :return: A dictionary with the following keys:
    :doc-author: Trelent
    """

    # Prepare input parameters
    symbol = symbol.upper()
    side = side.lower()
    amount_in = amount_in.lower()
    timeInForce = timeInForce.lower()
    bp = float(bp)
    # Configure quantity_or_price and amountIn based on the side of the transaction
    if side == 'buy':
        amountIn = 'dollars'
        quantity_or_price = float(quantity_or_price)

        # only use 1% of buying power or the next number of dollars up to the minimum order amount for the coin
        if bp is None:
            # Fetch buying power from robinhood
            profile = r.profiles.load_account_profile()
            os.environ['PROFILE'] = profile  # set an env variable for the profile
            bp = float(r.profiles.load_account_profile(info='buying_power'))  # get the buying power from the profile

        # Calculate quantity_or_price based on buying power
        quantity_or_price = bp * 0.01

        # Get the minimum order amount for the coin
        minimum_order_amount = minimum_orders_coins[symbol]

        # Adjust quantity_or_price if it's less than the minimum order amount
        if quantity_or_price < minimum_order_amount:
            quantity_or_price = minimum_order_amount

    else:
        amountIn = 'quantity'

    # Prepare quantity_or_price for the order_crypto call
    quantity_or_price = str(quantity_or_price)

    try:
        print(f'Attempting a {side} order of {quantity_or_price} {symbol}...')

        # Execute the order using robin stocks
        r.orders.order_crypto(
            symbol=str(symbol),
            quantityOrPrice=float(quantity_or_price),
            amountIn=str(amount_in),  # either 'quantity' or 'dollars'
            side=str(side),
            timeInForce='gtc',
            jsonify=True
        )

        print(f'{side.capitalize()}ing {quantity_or_price} {symbol}...')
        print(Fore.GREEN + f'Passed try statement in order! {side}' + Fore.RESET)
        time.sleep(1)  # sleep for 1 second
    except Exception as e:
        raise e

#### END ACTION FUNCTIONS ####

def login_setup():
    """
    The login_setup function is used to login to the robinhood account and get the account details.

    :return: The account_details_df and the login
    :doc-author: Trelent
    """

    # this is where we will setup the accounts details df and login to robinhood
    with open('config/credentials.json') as f:
        credentials = json.load(f)
    # login to robinhood
    login = r.login(credentials['username'], credentials['password'])
    # get the account details
    account_details = r.profiles.load_account_profile(info=None)
    # get the account details as a df
    account_details_df = pd.DataFrame(account_details, index=[0])

    #! Set up the logs
    # create a logs folder if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    # create a log file
    logging.basicConfig(filename='logs/robinhood.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.info('Started')


    return account_details_df, login

def robin_getter(coin):
    """
    The robin_getter function is used to get the available crypto currencies, historical data and current price of a given coin.
        Args:
            coin (str): The name of the cryptocurrency you want to get information on.

    :param coin: Get the data for that coin
    :return: A tuple of 3 items
    :doc-author: Trelent
    """


    # this is where we will get the data from robinhood
    # as we are logged in, list the crypto we can buy and sell
    crypto_available_on_robinhood = r.crypto.get_crypto_currency_pairs()
    # get the historical data for the crypto
    crypto_historicals = r.crypto.get_crypto_historicals(str(coin), "5minute", "day")
    # get the current price of the crypto should be the latest in the historicals
    crypto_price = r.crypto.get_crypto_quote(str(coin)) #todo check this is the latest price or is it ok to take the last price in the historicals


    return crypto_available_on_robinhood, crypto_historicals, crypto_price #note: should crypto_price be a df or a float?

#^getting the crypto position for a coin
@sleep_and_retry
def get_crypto_positions_in_account():
    # Get all crypto positions
    crypto_positions = r.crypto.get_crypto_positions()

    # Initialize an empty dictionary to store the positions
    positions_dict = {}

    # Iterate over the positions and store the coin information in the dictionary
    for position in crypto_positions:
        # Get the coin symbol
        symbol = position['currency']['code']

        # Get the quantity of the coin
        quantity = float(position['quantity_available'])

        # Get the average buy price of the coin
        #!average_buy_price = float(position['average_buy_price'])

        # Store the coin information in the dictionary
        positions_dict[symbol] = {
            'quantity': quantity
            #!'average_buy_price': average_buy_price
        }

    return positions_dict



def calculate_ta_indicators():
    """
    The calculate_ta_indicators function is the main function that calls all of the other functions.
    It will call robin_getter to get data from Robinhood, then it will call signal_engine to calculate signals for each coin, and finally it will call trading_function which takes actions based on those signals.

    :return: The following:
    :doc-author: Trelent
    """

    # this is the main module that will call the other modules
    coins_list = ['BTC', 'ETH', 'ADA', 'DOGE', 'MATIC', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'SOL', 'AVAX', 'XLM', 'BCH', 'XTZ']
    # fill the minimum_orders_coins dict with the minimum order amount for each coin
    minimum_orders_coins = pd.DataFrame()
    holdings_df = pd.DataFrame()

    # Usage
    crypto_positions_dict = get_crypto_positions_in_account()
    crypto_positions_df = pd.DataFrame(crypto_positions_dict) #todo check this is the correct way to do this
    #### print the crypto_positions_dict
    # populate the holdings_df with the crypto_positions_dict
    crypto_I_own = {}
    for key, value in crypto_positions_dict.items():
        holdings_df[key] = float(value['quantity'])
        crypto_I_own[key] = float(value['quantity'])




    # save this result to a json named `r_crypto_get_crypto_positions.json`
    # if data folder doesn't exist, create it
    if not os.path.exists('data'):
        os.makedirs('data')
    # save the crypto_I_own to a json file
    with open('data/crypto_I_own.json', 'w') as fp:
        json.dump(crypto_I_own, fp)

    # create the crypto_positions_df from coins_I_own and the crypto_positions_df from robinhood
    crypto_positions_df = pd.DataFrame(crypto_I_own, index=[0])



    # the min buy ammt is found at `currency`>`increment`
    # the amount we own is found at `currency`>`quantity_available`
    print(f'Getting the minimum order amount for each coin...')
    index_of_coin = 0
    for coin in tqdm(coins_list):
        tqdm.write(f'Getting the minimum order amount for {coin}...')
        coin_info = r.crypto.get_crypto_info(coin)
        coin_info_df = pd.DataFrame(coin_info, index=[index_of_coin])
        minimum_orders_coins = minimum_orders_coins.append(coin_info_df)
        index_of_coin += 1


    # this is where we will call the robin_getter
    coin_historicals_dfs = [] # this holds the historical data for each coin
    for coin in tqdm(coins_list):
        tqdm.write(f'Getting the historical data for {coin}...')
        crypto_available_on_robinhood, crypto_historicals, crypto_price = robin_getter(coin)
        coin_historicals_dfs.append(crypto_historicals)
        # print(Fore.BLUE + '>> debug: crypto_available_on_robinhood: ', crypto_historicals)
        print(Fore.BLUE + '>> debug: crypto_available_on_robinhood: ', type(crypto_historicals))
        crypto_data = {
            "coin": coin,
            "crypto_available_on_robinhood": crypto_available_on_robinhood,
            "crypto_historicals": crypto_historicals,
            "coin_mark_price": crypto_price['mark_price'],
            "coin_ask_price": crypto_price['ask_price'],
            "coin_bid_price": crypto_price['bid_price'],
            "coin_high_price": crypto_price['high_price'],
            "coin_low_price": crypto_price['low_price'],
            "coin_open_price": crypto_price['open_price']
        }
        # add the crypto_data to the holdings_df
        holdings_df[coin] = crypto_data
        # print(Fore.BLUE + '>> debug: crypto_data: ', crypto_data)
        # print(Fore.BLUE + '>> debug: holdings_df: ', holdings_df)
        # print(Fore.BLUE + '>> debug: holdings_df: ', type(holdings_df))
        # print(Fore.BLUE + '>> debug: holdings_df: ', holdings_df[coin])
        # print(Fore.BLUE + '>> debug: holdings_df: ', type(holdings_df[coin]))
    # save this result to a json named `r_crypto_get_crypto_positions.json`
    # if data folder doesn't exist, create it
    if not os.path.exists('data'):
        os.makedirs('data')
    # save the crypto_I_own to a json file
    with open('data/crypto_I_own.json', 'w') as fp:
        json.dump(crypto_I_own, fp)

    # save the holdings_df to a json file
    with open('data/holdings_df.json', 'w') as fp:
        json.dump(holdings_df, fp)

    # save the minimum_orders_coins to a json file
    with open('data/minimum_orders_coins.json', 'w') as fp:
        json.dump(minimum_orders_coins, fp)

    # this is where we will call the signal_engine
    logging.info('Calculating signals...')

    signals_dict = {}

    for df in coin_historicals_dfs:
        crypto_historicals = df #todo check this is the correct df
        crypto_historicals_df = pd.DataFrame(crypto_historicals)
        if 'USD' in str(crypto_historicals_df['symbol'][0]):
            coin = str(crypto_historicals_df['symbol'][0])[:3] # remove the '-USD' from the coin name
        elif '-USD' in str(crypto_historicals_df['symbol'][0]):
            coin = str(crypto_historicals_df['symbol'][0])[:4]
        elif 'DOGE' in str(crypto_historicals_df['symbol'][0]):
            coin = str("DOGE") #todo -- this is a hack, need to fix this
        else:
            coin = str(crypto_historicals_df['symbol'][0])

        logging.info('  Calculating signals for {}...'.format(coin))
        buy_signal, sell_signal, hold_signal = signal_engine(df, coin)
        signals_dict[coin] = [buy_signal, sell_signal, hold_signal]
    # this is where we will call the trading_function
    for coin in tqdm(coins_list):
        if coin == 'DOGE':
            continue
        try:
            # get the signals for the coin
            buy_signal = signals_dict[coin][0]
            sell_signal = signals_dict[coin][1]
            hold_signal = signals_dict[coin][2]

            # if the buy signal is greater than the sell signal and the hold signal then buy the coin with 1% of the buying_power OR the next degree up from that value in the minimum order increment size for that coin
            if buy_signal > sell_signal and buy_signal > hold_signal:
                # buy the coin
                order_crypto(symbol=coin,
                                quantity_or_price=0.01* float(holdings_df[coin]),
                                amount_in='dollars',
                                side='buy',
                                timeInForce='gtc')
                print(f'Buying {coin}...')
                time.sleep(1)
            # if the sell signal is greater than the buy signal and the hold signal then sell the coin with 1% of the buying_power OR the next degree up from that value in the minimum order increment size for that coin
            elif sell_signal > buy_signal and sell_signal > hold_signal:

                # check if we have enough of the coin to sell
                if float(holdings_df[coin]) < 0.01 or coin not in holdings_df.columns:
                    print(f'Not enough {coin} to sell...')
                    continue

                # sell the coin
                order_crypto(symbol=str(coin),
                                quantity_or_price=0.80 * float(holdings_df[coin]),
                                amount_in='dollars',
                                side='sell',
                                timeInForce='gtc')
                print(f'Selling {coin}...')
                time.sleep(1)
            # if the hold signal is greater than the buy signal and the sell signal then do nothing
            elif hold_signal > buy_signal and hold_signal > sell_signal:
                print(f'Hold {coin}...')
            # if the buy signal is equal to the sell signal and the hold signal then do nothing
            elif buy_signal == sell_signal and buy_signal == hold_signal:
                print(f'Hold {coin}... as buy_signal == sell_signal == hold_signal')
            # if the buy signal is equal to the sell signal then do nothing
            elif buy_signal == sell_signal:
                print(f'Hold {coin}... as buy_signal == sell_signal')
            else:
                print(f'Hold {coin}... \n buy_signal: {buy_signal} \n sell_signal: {sell_signal} \n hold_signal: {hold_signal}')
        except Exception as e:
            # show the traceback
            traceback.print_exc()
            logging.info(' {} is the error'.format(e))
            print(f'Hold {coin}... \n buy_signal: {buy_signal} \n sell_signal: {sell_signal} \n hold_signal: {hold_signal}')
            logging.info('  {} buy signal: {}'.format(coin, buy_signal))
            logging.info('  {} sell signal: {}'.format(coin, sell_signal))
            logging.info('  {} hold signal: {}'.format(coin, hold_signal))
            continue

def signal_engine(df, coin):
    # this is where we will calculate the signals for buys and sells for each coin
    # assume we are already logged in
    # using finta to calculate the signals
    # using robin_stocks to execute the buys and sells

    # cast the inputs to the correct formats first
    df = pd.DataFrame(df)
    coin = str(coin) # make sure the coin is a string

    # filter the df to only include the columns we need
    df = df[['begins_at', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']]
    # rename the columns to be compatible with finta
    df = df.rename(columns={'begins_at': 'date', 'open_price': 'open', 'close_price': 'close', 'high_price': 'high', 'low_price': 'low', 'volume': 'volume'})
    # also make sure the coin is in the df
    df['coin'] = coin

    # cast all the values in these columns to floats
    df['date'] = df['date'].astype(str)
    logging.info('  df date: {}'.format(df['date']))
    logging.info('casting df data to floats...')
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # define the point system for buys and sells, give each signal a score
    buy_signal = 0
    sell_signal = 0
    hold_signal = 0
    # calculate the signals
    """
    The TA.RSI function expects a pandas DataFrame as input, with columns named 'open', 'high', 'low', and 'close'. The DataFrame should contain the price data for the asset you're analyzing.
    """
    rsi = TA.RSI(df[['open', 'high', 'low', 'close']]).to_list()[-1]
    macd = TA.MACD(df)['MACD'].to_list()[-1]
    macd_signal = TA.MACD(df)['SIGNAL'].to_list()[-1]
    upper_bollingerband = TA.BBANDS(df)[
        'BB_UPPER'
        ].to_list()[-1]
    lower_bollingerband = TA.BBANDS(df)[
        'BB_LOWER'
        ].to_list()[-1]
    ma200 = TA.SMA(df, 200).to_list()[-1]
    ma50 = TA.SMA(df, 50).to_list()[-1]
    current_price = df['close'].iloc[-1]
    # todo: add the logic for the buys and sells
    # Buy if the Relative Strength Index (RSI) is below 30 and the Moving Average Convergence Divergence (MACD) is above 0. RSI values below 30 often indicate an oversold condition, suggesting the asset may be undervalued. A positive MACD suggests bullish (upward) momentum.
    if rsi < 30 and macd > 0:
        # buy
        # order_crypto(coin, 10, amount_in='dollars', side='buy')
        # print(f'Buying {coin} at ${current_price}...')
        # logging.info(f'Buying {coin} at ${current_price}... because RSI is {rsi} and MACD is {macd}, this is a buy signal')
        #^ instead of buying for real,  increment the buy_signal by 1
        buy_signal += 1
    # Sell if the RSI is above 70 and MACD is below 0. RSI values above 70 often indicate an overbought condition, suggesting the asset may be overvalued. A negative MACD suggests bearish (downward) momentum.
    elif rsi > 70 and macd < 0:
        # sell
        # order_crypto(coin, 10, amount_in='dollars', side='sell')
        # print(f'Selling {coin} at ${current_price}...')
        # logging.info(f'Selling {coin} at ${current_price}... because RSI is {rsi} and MACD is {macd}, this is a sell signal')
        #^ instead of selling for real,  increment the sell_signal by 1
        sell_signal += 1
    # Buy if the MACD is greater than its signal line and the current price is greater than the Simple Moving Average (SMA). This may suggest the asset's price has upward momentum and is trading above its average price.
    elif macd > macd_signal and current_price > ma200:
        # buy
        # order_crypto(coin, 10, amount_in='dollars', side='buy')
        # print(f'Buying {coin} at ${current_price}...')
        # logging.info(f'Buying {coin} at ${current_price}... because MACD is {macd} and MACD Signal is {macd_signal}, this is a buy signal')
        #^ instead of buying for real,  increment the buy_signal by 1
        buy_signal += 1
    # Sell if the MACD is less than its signal line and the current price is less than the SMA. This may suggest the asset's price has downward momentum and is trading below its average price.
    elif macd < macd_signal and current_price < ma200:
        # sell
        # order_crypto(coin, 10, amount_in='dollars', side='sell')
        # print(f'Selling {coin} at ${current_price}...')
        # logging.info(f'Selling {coin} at ${current_price}... because MACD is {macd} and MACD Signal is {macd_signal}, this is a sell signal')
        #^ instead of selling for real,  increment the sell_signal by 1
        sell_signal += 1
    # Buy if the current price is below the lower Bollinger Band. This may suggest the asset is oversold and due for a price correction.
    elif current_price < lower_bollingerband:
        # buy
        # order_crypto(coin, 10, amount_in='dollars', side='buy')
        # print(f'Buying {coin} at ${current_price}...')
        # logging.info(f'Buying {coin} at ${current_price}... because current price is {current_price} and lower bollinger band is {lower_bollingerband}, this is a buy signal')
        #^ instead of buying for real,  increment the buy_signal by 1
        buy_signal += 1
    # Sell if the current price is above the upper Bollinger Band. This may suggest the asset is overbought and due for a price correction.
    elif current_price > upper_bollingerband:
        # sell
        # order_crypto(coin, 10, amount_in='dollars', side='sell')
        # print(f'Selling {coin} at ${current_price}...')
        # logging.info(f'Selling {coin} at ${current_price}... because current price is {current_price} and upper bollinger band is {upper_bollingerband}, this is a sell signal')
        #^ instead of selling for real,  increment the sell_signal by 1
        sell_signal += 1
    # Buy if the current price is above the 200-day SMA and the 50-day SMA is above the 200-day SMA. This may suggest the asset's price is in an uptrend and is trading above its average price.
    elif current_price > ma200 and ma50 > ma200:
        # buy
        # order_crypto(coin, 10, amount_in='dollars', side='buy')
        # print(f'Buying {coin} at ${current_price}...')
        # logging.info(f'Buying {coin} at ${current_price}... because current price is {current_price} and 200-day SMA is {ma200}, this is a buy signal')
        #^ instead of buying for real,  increment the buy_signal by 1
        buy_signal += 1
    # Sell if the current price is below the 200-day SMA and the 50-day SMA is below the 200-day SMA. This may suggest the asset's price is in a downtrend and is trading below its average price.
    elif current_price < ma200 and ma50 < ma200:
        # sell
        # order_crypto(coin, 10, amount_in='dollars', side='sell')
        # print(f'Selling {coin} at ${current_price}...')
        # logging.info(f'Selling {coin} at ${current_price}... because current price is {current_price} and 200-day SMA is {ma200}, this is a sell signal')
        #^ instead of selling for real,  increment the sell_signal by 1
        sell_signal += 1
    # Buy if the current price is above the 200-day SMA and the 50-day SMA is below the 200-day SMA. This may suggest the asset's price is in a consolidation phase and is trading above its average price.
    elif current_price > ma200 and ma50 < ma200:
        # buy
        # order_crypto(coin, 10, amount_in='dollars', side='buy')
        # print(f'Buying {coin} at ${current_price}...')
        # logging.info(f'Buying {coin} at ${current_price}... because current price is {current_price} and 200-day SMA is {ma200}, this is a buy signal')
        #^ instead of buying for real,  increment the buy_signal by 1
        buy_signal += 1
    # Sell if the current price is below the 200-day SMA and the 50-day SMA is above the 200-day SMA. This may suggest the asset's price is in a consolidation phase and is trading below its average price.
    elif current_price < ma200 and ma50 > ma200:
        # sell
        # order_crypto(coin, 10, amount_in='dollars', side='sell')
        # print(f'Selling {coin} at ${current_price}...')
        # logging.info(f'Selling {coin} at ${current_price}... because current price is {current_price} and 200-day SMA is {ma200}, this is a sell signal')
        #^ instead of selling for real,  increment the sell_signal by 1
        sell_signal += 1
    else:
        hold_signal += 1
    # Print the signals as a sanity check
    # show them like this:
    # (+): |||
    # (-): |||||
    # (0): ||||
    # print(f'(+): {buy_signal} ',end='')
    # print('|'*int(buy_signal))
    # print("")
    # print(f'(-): {sell_signal} ',end='')
    # print('|'*int(sell_signal))
    # print("")
    # print(f'(0): {hold_signal} ',end='')
    # print('|'*int(hold_signal))
    # print("")
    print(f'(+): {buy_signal} (-): {sell_signal} (0): {hold_signal}')

    return buy_signal, sell_signal, hold_signal

def trading_function():
    """
    The trading_function function is the main function that executes all of the buys and sells.
    It uses order_crypto() to execute the buys and sells.
    It always buys with 1% of current buying power (amount_in='dollars').
    It always sells with 100% of current position (amount_in='amount').
    The side parameter can be set to 'buy' or 'sell' to specify which action should be taken.  The symbol parameter specifies which coin should be bought or sold, while quantity_or_price specifies how much money should go into each buy order, or how many coins are in each sell order

    :return: A dictionary of the positions after executing the buys and sells
    :doc-author: Trelent
    """
    # this is where we will execute the buys and sells
    # use order_crypto() to execute the buys and sells
    # always buy with 1% of current buying power (amount_in='dollars')
    # always sell with 100% of current position (amount_in='amount')
    # use the side='buy' or side='sell' to specify the action
    # use the symbol=coin to specify the coin
    # use quantity_or_price=0.01 * float(buying_power) to specify the amount to buy
    # use quantity_or_price=position to specify the amount to sell
    # use info='buying_power' to get the current buying power
    # use info='quantity' to get the current position

    # get the current buying power
    buying_power = r.profiles.load_account_profile(info='buying_power')
    # cancel all open crypto orders
    try:
        r.orders.cancel_all_crypto_orders()
        time.sleep(10)
    except Exception as e:
        print(e)
        print('Unable to cancel orders...')
        logging.info(f'Unable to cancel orders...{e}')
    print(f'Buying power is {buying_power}')
    # iterate over the coins
    for coin in tqdm(signals_dict.keys(), total=len(signals_dict.keys())):
        # get the signals for the coin
        buy_signal = signals_dict[coin][0]
        sell_signal = signals_dict[coin][1]
        hold_signal = signals_dict[coin][2]

        # if there is any sell signal then cancel all orders for the coin first to prevent any open orders from conflicting with the sell order
        if sell_signal > 0:
            try:
                r.orders.cancel_all_crypto_orders(symbol=coin)
            except Exception as e:
                print(e)
                print(f'Unable to cancel orders for {coin}...')
                logging.info(f'Unable to cancel orders for {coin}...{e}')

        # get the current position for the coin
        position = r.crypto.get_crypto_positions(info='quantity')

        # if the buy signal is greater than the sell signal and the hold signal then buy the coin with 1% of the buying_power
        if buy_signal > sell_signal and buy_signal > hold_signal:
            # buy the coin
            order_crypto(symbol=coin,
                         quantity_or_price=0.01 * float(buying_power),
                         amount_in='dollars',
                         side='buy',
                         bp = buying_power,
                         timeInForce='gtc')
            # update bp (buying power)
            buying_power -= 0.01 * float(buying_power)

        # if the sell signal is greater than the buy signal and the hold signal then sell the coin with 100% of the current position
        elif sell_signal > buy_signal and sell_signal > hold_signal:
            # sell the coin
            order_crypto(symbol=coin,
                         quantity_or_price=float(position),
                         amount_in='amount',
                         side='sell',
                         bp = buying_power,
                         timeInForce='gtc')




def record_keeping_engine(coin, cost, quantity, side, current_price, buy_signal, sell_signal, hold_signal):

    # cast the inputs to their proper types
    coin = str(coin)
    cost = float(cost)
    quantity = float(quantity)
    side = str(side)
    current_price = float(current_price)
    buy_signal = int(buy_signal) #note: this will throw an error if buy_signal is a float, so we need to cast it to an int
    sell_signal = int(sell_signal)
    hold_signal = int(hold_signal)


    # this is where we will record the buys and sells to a csv file named `coin_trades.csv` with headers `coin`, `cost`, `quantity`, `side`, `current_price`, `buy_signal`, `sell_signal`, `hold_signal`
    logging.info(f'Writing to coin_trades.csv...')
    # open the csv file
    with open('coin_trades.csv', 'w') as csvfile:
        # create the writer
        writer = csv.writer(csvfile)
        # write the headers
        writer.writerow(['coin', 'cost', 'quantity', 'side', 'current_price', 'buy_signal', 'sell_signal', 'hold_signal'])
        # write the data
        writer.writerow([coin, cost, quantity, side, current_price, buy_signal, sell_signal, hold_signal])
    logging.info(f'Wrote to coin_trades.csv')

def is_daytime():
    # return true if it is daytime and false if it is nighttime in CST
    # get the current time in CST
    current_time = datetime.now(timezone('US/Central'))
    # get the current hour
    current_hour = current_time.hour
    # if the current hour is between 8am and 8pm then return true
    if current_hour >= 8 and current_hour <= 20:
        return True
    # otherwise return false
    else:
        return False
## Main
if __name__ == '__main__':
    login_setup()
    while True:
        # run brain module
        calculate_ta_indicators()
        # run action engine
        trading_function()
        # run record keeping engine
        #record_keeping_engine(coin, cost, quantity, side, current_price, buy_signal, sell_signal, hold_signal)

        # sleep for 5 minutes if daytime or 30 minutes if nighttime
        if is_daytime():
            time.sleep(300)
        else:
            print('Sleeping for 30 minutes...')
            time.sleep(1800)