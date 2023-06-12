from finta import TA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import traceback
from robin_stocks import robinhood as r
import json
import time
import csv
from time import sleep
import os
from ratelimit import limits, sleep_and_retry
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
        print(f'{side.capitalize()}ing {quantity_or_price} {symbol}...')
        time.sleep(1) # sleep for 1 second
    except Exception as e:
        raise e

#### END ACTION FUNCTIONS ####

def login_setup():
    # this is where we will setup the accounts details df and login to robinhood
    with open('credentials.json') as f:
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
    # this is where we will get the data from robinhood
    # as we are logged in, list the crypto we can buy and sell
    crypto_available = r.crypto.get_crypto_currency_pairs()
    # get the historical data for the crypto
    crypto_historicals = r.crypto.get_crypto_historicals(str(coin), "5minute", "day")
    # get the current price of the crypto should be the latest in the historicals
    crypto_price = r.crypto.get_crypto_quote(str(coin)) #todo check this is the latest price or is it ok to take the last price in the historicals
    return crypto_available, crypto_historicals, crypto_price #note: should crypto_price be a df or a float?

def brain_module():
    # this is the main module that will call the other modules
    coins_list = ['BTC', 'ETH', 'ADA', 'DOGE', 'MATIC', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'SOL', 'AVAX', 'XLM', 'BCH', 'XTZ']
    # this is where we will call the robin_getter
    coins_dfs = []
    for coin in coins_list:
        crypto_available, crypto_historicals, crypto_price = robin_getter(coin)
        coins_dfs.append(crypto_historicals)
    # this is where we will call the signal_engine
    logging.info('Calculating signals...')
    signals_dict = {}
    for df in coins_dfs:
        coin = str(df['symbol'][0])
        buy_signal, sell_signal, hold_signal = signal_engine(df, coin)
        signals_dict[coin] = [buy_signal, sell_signal, hold_signal]
        logging.info('  {} buy signal: {}'.format(coin, buy_signal))
        logging.info('  {} sell signal: {}'.format(coin, sell_signal))
        logging.info('  {} hold signal: {}'.format(coin, hold_signal))

    # this is where we will call the action_engine
    for coin in coins_list:

        # get the signals for the coin
        buy_signal = signals_dict[coin][0]
        sell_signal = signals_dict[coin][1]
        hold_signal = signals_dict[coin][2]

        # if the buy signal is greater than the sell signal and the hold signal then buy the coin with 1% of the buying_power OR the next degree up from that value in the minimum order increment size for that coin
        if buy_signal > sell_signal and buy_signal > hold_signal:
            # buy the coin
            order_crypto(symbol=coin,
                            quantity_or_price=0.01,
                            amount_in='dollars',
                            side='buy')
        # if the sell signal is greater than the buy signal and the hold signal then sell the coin with 1% of the buying_power OR the next degree up from that value in the minimum order increment size for that coin
        elif sell_signal > buy_signal and sell_signal > hold_signal:
            # sell the coin
            order_crypto(symbol=coin,
                            quantity_or_price=0.01,
                            amount_in='dollars',
                            side='sell')
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

def signal_engine(df, coin):
    # this is where we will calculate the signals for buys and sells for each coin
    # assume we are already logged in
    # using finta to calculate the signals
    # using robin_stocks to execute the buys and sells

    # filter the df to only include the columns we need
    df = df[['begins_at', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']]
    # also make sure the coin is in the df
    df['coin'] = coin

    # define the point system for buys and sells, give each signal a score
    buy_signal = 0
    sell_signal = 0
    hold_signal = 0
    # calculate the signals
    rsi = TA.RSI(df)
    macd = TA.MACD(df)
    macd_signal = TA.MACD(df)['MACD_SIGNAL']
    upper_bollingerband = TA.BBANDS(df)['BB_UPPER']
    lower_bollingerband = TA.BBANDS(df)['BB_LOWER']
    ma200 = TA.SMA(df, 200)
    ma50 = TA.SMA(df, 50)
    ma20 = TA.SMA(df, 20)
    current_price = df['close_price'].iloc[-1]
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
    print(f'(+): {buy_signal} ',end='')
    print('|'*int(buy_signal))
    print("")
    print(f'(-): {sell_signal} ',end='')
    print('|'*int(sell_signal))
    print("")
    print(f'(0): {hold_signal} ',end='')
    print('|'*int(hold_signal))
    print("")
    print(f'(+): {buy_signal} (-): {sell_signal} (0): {hold_signal}')

    return buy_signal, sell_signal, hold_signal

def action_engine():
    # this is where we will execute the buys and sells
    pass

def record_keeping_engine(coin, cost, quantity, side, current_price, buy_signal, sell_signal, hold_signal):
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

## Main
if __name__ == '__main__':
    login_setup()
    # run brain module
    brain_module()
    # run action engine
    action_engine()
    # run record keeping engine
    record_keeping_engine()
