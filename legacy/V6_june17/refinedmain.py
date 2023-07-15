from finta import TA
from colorama import Fore, Back, Style
from datetime import datetime, timedelta
import datetime
import matplotlib.pyplot as plt, pandas as pd, numpy as np
import logging, traceback, json, time, csv, os
from robin_stocks import robinhood as r
import statistics
from icecream import ic
from pytz import timezone
from tqdm import tqdm
import random
import os
from time import sleep
from ratelimit import limits, sleep_and_retry
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime
from pytz import timezone
from colorama import Fore, Back, Style
import asyncio
from tqdm import tqdm
from datetime import datetime
from pytz import timezone
tracking_dict = {
    'BTC':{
        'last_buyin_price': 0.0,
        'last_buyin_usd': 0.0,
        'trigger_stoploss_price': 0.0,
        'trigger_stoploss_coin_pct': 0.0,
    }
}
signals_dict = {}
minimum_orders_coins = {}
import ast
import re
import pandas as pd
import asyncio
PERCENTAGE_IN_PLAY = 0.40
ticking_iterator = 0
percentage_in_play = PERCENTAGE_IN_PLAY
loop_count = 0
RESET = False
stop_loss_percent = 0.05
verboseMode = True
BUYING_POWER = 0.0
TOTAL_CRYPTO_DOLLARS = 0.0
threshold_total_crypto_per_coin = 0.10
crypto_I_own = {}
@sleep_and_retry
def order_crypto(symbol, quantity_or_price, amount_in='dollars', side='buy', bp=None, timeInForce='gtc'):
    """
    The order_crypto function is used to place a buy or sell order for a given crypto currency.
    :param symbol: Specify the crypto symbol you want to trade
    :param quantity_or_price: Determine the amount of crypto to buy or sell
    :param amount_in: Specify whether the quantity_or_price parameter is in dollars or shares
    :param side: Determine whether to buy or sell
    :param bp: Set the buying power for a buy order
    :param timeInForce: Specify the duration of the order
    :return: A dict with the following keys:
    :doc-author: Trelent
    """
    global BUYING_POWER
    if symbol is None:
        return
    symbol = symbol.upper() if type(symbol) == str else str(symbol).upper()
    side = side.lower() if type(side) == str else str(side).lower()
    try:
        amount_in = amount_in.lower() if side == 'sell' else 'dollars'
        timeInForce = timeInForce.lower()
    except Exception as e:
        amount_in = float(amount_in)
        timeInForce = 'gtc'
    if side == 'buy':
        profile = r.profiles.load_account_profile()
        os.environ['PROFILE'] = str(profile)
        if bp is None:
            bp = BUYING_POWER * percentage_in_play
        if symbol == 'DOG': symbol = 'DOGE'
        quantity_or_price=max(1.05,0.10 * float(BUYING_POWER))
        if quantity_or_price > BUYING_POWER:
            if verboseMode:
                print(Fore.RED + f'Not enough buying power to buy {quantity_or_price} {symbol}...' + Fore.RESET)
            return
        time.sleep(random.randint(1, 3))
    try:
        r.orders.order_crypto(symbol=str(symbol), quantityOrPrice=quantity_or_price, amountIn=amount_in, side=side, timeInForce='gtc', jsonify=True)
        time.sleep(random.randint(1, 3))
    except Exception as e:
        print(Fore.RED + f'Failed try statement in order! {side} because \n{e}' + Fore.RESET)
    if side == 'sell':
        if verboseMode:
            current_price = float(r.crypto.get_crypto_quote(symbol)['mark_price'])
            print(Fore.RED + f'Sell order for {quantity_or_price} {symbol} worth ${quantity_or_price*current_price}' + Fore.RESET)
        pass
    else:
        if verboseMode:
            print(Fore.GREEN + f'Buy order at ${quantity_or_price} of {symbol}' + Fore.RESET)
        pass
    return
def login_setup():
    """
    The login_setup function is used to log into the Robinhood API and return a dataframe of account details.
    It also sets up logging for debugging purposes.
    :return: A dataframe and a login object
    :doc-author: Trelent
    """
    global BUYING_POWER
    with open('config/credentials.json') as f:
        credentials = json.load(f)
    login = r.login(credentials['username'], credentials['password'])
    account_details = r.profiles.load_account_profile(info=None)
    if isinstance(account_details, dict):
        account_details_df = pd.DataFrame(account_details, index=[0])
    else:
        print(Fore.BLUE + "DEBUGGING -> SETTING account_details to a dictionary" + Fore.RESET)
        account_details_df = pd.DataFrame(account_details.to_dict(), index=[0])
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(filename='logs/robinhood.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.info('Started')
    BUYING_POWER = float(account_details_df['onbp'])
    return account_details_df, login
@sleep_and_retry
def robin_getter(coin):
    """
    The robin_getter function is a function that takes in a coin as an argument and returns the following:
        1. A list of all crypto currencies available on Robinhood
        2. The historicals for the given coin, with 5 minute intervals, over 24 hours (24_7)
        3. The current price of the given coin
    :param coin: Specify which coin you want to get the data for
    :return: A tuple of three items
    :doc-author: Trelent
    """
    crypto_available_on_robinhood = r.crypto.get_crypto_currency_pairs()
    crypto_historicals = r.crypto.get_crypto_historicals(str(coin), "5minute", "day", "24_7", info=None)
    crypto_price = r.crypto.get_crypto_quote(str(coin))
    return crypto_available_on_robinhood, crypto_historicals, crypto_price
@sleep_and_retry
def get_crypto_positions_in_account():
    """
    The get_crypto_positions_in_account function returns a dictionary of the crypto positions in your account.
    The keys are the symbols, and the values are dictionaries with one key: quantity.
    :return: A dictionary of all crypto positions in the account
    :doc-author: Trelent
    """
    crypto_positions = r.crypto.get_crypto_positions()
    positions_dict = {}
    for position in crypto_positions:
        symbol = position['currency']['code']
        quantity = float(position['quantity_available'])
        positions_dict[symbol] = {
            'quantity': quantity
        }
    return positions_dict
def resetter():
    """
    The resetter function is used to reset the crypto portfolio.
    It does this by selling all positions of crypto and deleting all orders.
    :return: A boolean value
    :doc-author: Trelent
    """
    if RESET:
        crypto_positions = r.crypto.get_crypto_positions()
        for position in crypto_positions:
            symbol = position['currency']['code']
            quantity = float(position['quantity_available'])
            order_crypto(symbol,
                        amount_in='quantity',
                        quantity_or_price=quantity,
                        side='sell')
        orders = r.orders.get_all_open_crypto_orders()
        for order in orders:
            if order['side'] == 'buy':
                r.orders.cancel_crypto_order(order['id'])
                print(f'Cancelled buy order {order["id"]}...')
async def log_file_size_checker():
    """
    The log_file_size_checker function is an async function that checks the size of the log file and removes lines from the start of the file to maintain a rolling log of 1000 lines.
    :return: None
    :doc-author: Trelent
    """
    while True:
        with open('logs/robinhood.log', 'r') as f:
            lines = f.readlines()
            if len(lines) > 1000:
                num_lines_to_remove = len(lines) - 1000
                with open('logs/robinhood.log', 'w') as f:
                    f.writelines(lines[num_lines_to_remove:])
        await asyncio.sleep(1200)
@sleep_and_retry
def get_account():
    """
    The get_account function returns the account information for a user.
    :return: A dictionary
    :doc-author: Trelent
    """
    account = r.profiles.load_account_profile(info=None)
    return account
def calculate_ta_indicators():
    """
    The calculate_ta_indicators function is the main function of this module. It does the following:
        1. Gets a list of coins to trade from `coins_list` variable
        2. Gets the minimum order amount for each coin in `coins_list` and saves it as a csv file called 'data/minimum_orders_coins'
        3. For each coin in `coins_list`, gets its historical data, price, and whether or not it's available on Robinhood (i.e., if you can buy/sell it)
            - Saves all that info into a dictionary called holdings dict which is
    :return: The global variable `crypto_signals`
    :doc-author: Trelent
    """
    global crypto_I_own
    global ticking_iterator
    global holdings_df
    global minimum_orders_coins
    global crypto_positions_df
    global BUYING_POWER
    coins_list = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']
    os.environ['COINS_LIST'] = str(coins_list)
    minimum_orders_coins = pd.DataFrame()
    holdings_df = pd.DataFrame()
    crypto_positions_dict = get_crypto_positions_in_account()
    crypto_positions_df = pd.DataFrame(crypto_positions_dict)
    crypto_I_own = {}
    for key, value in crypto_positions_dict.items():
        holdings_df[key] = float(value['quantity'])
        crypto_I_own[key] = float(value['quantity'])
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/crypto_I_own.json', 'w') as fp:
        json.dump(crypto_I_own, fp)
    crypto_positions_df = pd.DataFrame(crypto_I_own, index=[0])
    if os.path.exists('data/minimum_orders_coins.csv'):
        minimum_orders_coins = pd.read_csv('data/minimum_orders_coins.csv')
        minimum_orders_coins = minimum_orders_coins.set_index('symbol')
    else:
        ic()
        index_of_coin = 0
        for coin in coins_list:
            ic()
            coin_info = r.crypto.get_crypto_info(coin)
            coin_info_df = pd.DataFrame(coin_info, index=[index_of_coin])
            minimum_orders_coins = minimum_orders_coins.append(coin_info_df)
            index_of_coin += 1
        minimum_orders_coins = minimum_orders_coins.set_index('symbol')
        minimum_orders_coins.to_csv('data/minimum_orders_coins.csv')
    coin_historicals_dfs = []
    for coin in tqdm(coins_list):
        crypto_available_on_robinhood, crypto_historicals, crypto_price = robin_getter(coin)
        coin_historicals_dfs.append(crypto_historicals)
        crypto_data = {
            "coin": coin,
            "crypto_available_on_robinhood": crypto_available_on_robinhood,
            "crypto_historicals": crypto_historicals,
            "coin_mark_price": float(crypto_price['mark_price']),
            "coin_ask_price":float(crypto_price['ask_price']),
            "coin_bid_price": float(crypto_price['bid_price']),
            "coin_high_price": float(crypto_price['high_price']),
            "coin_low_price": float(crypto_price['low_price']),
            "coin_open_price": float(crypto_price['open_price'])
        }
        holdings_df[coin] = crypto_data
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/crypto_I_own.json', 'w') as fp:
        ic()
        json.dump(crypto_I_own, fp)
    with open('data/holdings_df.json', 'w') as fp:
        holdings_dict = holdings_df.to_dict()
        holdings_json = json.dumps(holdings_dict)
        fp.write(holdings_json)
    logging.info('Calculating signals...')
    signals_dict = {}
    for df in coin_historicals_dfs:
        crypto_historicals = df
        crypto_historicals_df = pd.DataFrame(crypto_historicals)
        if 'USD' in str(crypto_historicals_df['symbol'][0]):
            coin = str(crypto_historicals_df['symbol'][0])[:3]
        elif '-USD' in str(crypto_historicals_df['symbol'][0]):
            coin = str(crypto_historicals_df['symbol'][0])[:4]
        elif 'DOGE' in str(crypto_historicals_df['symbol'][0]):
            coin = str("DOGE")
        else:
            coin = str(crypto_historicals_df['symbol'][0])
        logging.info('  Calculating signals for {}...'.format(coin))
        buy_signal, sell_signal, hold_signal = signal_engine(df, coin)
        signals_dict[coin] = [buy_signal, sell_signal, hold_signal]
    for coin in coins_list:
        if coin not in signals_dict.keys():
            signals_dict[coin] = [0, 0, 0]
        if coin == 'DOGE':
            continue
        try:
            if isinstance(signals_dict[coin][0], str):
                signals_dict[coin][0] = int(signals_dict[coin][0])
            if isinstance(signals_dict, str):
                signals_dict[coin][1] = int(signals_dict[coin][1])
            signals_dict = pd.DataFrame(signals_dict)
            buy_signal = signals_dict[coin][0] if type(signals_dict[coin][0]) == int else int(signals_dict[coin][0])
            sell_signal = signals_dict[coin][1] if type(signals_dict[coin][1]) == int else int(signals_dict[coin][1])
            hold_signal = signals_dict[coin][2] if type(signals_dict[coin][2]) == int else int(signals_dict[coin][2])
            if buy_signal > sell_signal and buy_signal > hold_signal:
                cryptoiownlist = crypto_I_own
                if type(crypto_I_own) == dict and coin in crypto_I_own.keys():
                    pass
                elif type(crypto_I_own) == str and coin in crypto_I_own:
                    if float(crypto_I_own[coin]) > 0.01:
                        pass
                else:
                    print(type(crypto_I_own))
                    print(crypto_I_own)
                    ic()
                order_crypto(symbol=str(coin),
                                quantity_or_price=max(1.00,0.10 * float(BUYING_POWER)),
                                amount_in='dollars',
                                side='buy',
                                timeInForce='gtc')
                iteration = 0
                BUYING_POWER -= max(1.00, 0.10 * BUYING_POWER)
            elif sell_signal > buy_signal and sell_signal > hold_signal:
                coins_I_own = crypto_I_own
                if type(coins_I_own) == str:
                    coins_I_own = ast.literal_eval(coins_I_own)
                else:
                    pass
                coins_owned = coins_I_own[coin]
                coins_owned = float(coins_owned)
                if coins_owned < 0.01 or coin not in coins_I_own.keys():
                    continue
                print(f'Selling {coin}...')
                order_crypto(symbol=str(coin),
                                quantity_or_price=0.80 * float(crypto_I_own[coin]),
                                amount_in='amount',
                                side='sell',
                                timeInForce='gtc')
                time.sleep(1)
            elif hold_signal > buy_signal and hold_signal > sell_signal:
                pass
            elif buy_signal == sell_signal and buy_signal == hold_signal:
                pass
            elif buy_signal == sell_signal:
                pass
            else:
                ic()
        except Exception as e:
            traceback.print_exc()
            logging.info(' {} is the error'.format(e))
            logging.info('  {} buy signal: {}'.format(coin, buy_signal))
            logging.info('  {} sell signal: {}'.format(coin, sell_signal))
            logging.info('  {} hold signal: {}'.format(coin, hold_signal))
            continue
    os.environ['CRYPTO_SIGNALS'] = str(signals_dict)
    global crypto_signals
    crypto_signals = signals_dict
def signal_engine(df, coin):
    """
    The signal_engine function takes in a dataframe and a coin name.
    It then performs technical analysis on the dataframe to determine if there is a buy, sell or hold signal.
    The function returns three values: buy_signal, sell_signal and hold_signal which are all either 0 or 1.
    :param df: Pass the dataframe of the coin we are analyzing
    :param coin: Determine which coin to buy
    :return: A buy signal, sell signal, and hold signal
    :doc-author: Trelent
    """
    global signals_dict
    global crypto_I_own
    global stop_loss_percent
    global TOTAL_CRYPTO_DOLLARS
    global BUYING_POWER
    global ticking_iterator
    global threshold_total_crypto_per_coin
    df = pd.DataFrame(df)
    coin = str(coin)
    df = df[['begins_at', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']]
    df = df.rename(columns={'begins_at': 'date', 'open_price': 'open', 'close_price': 'close', 'high_price': 'high', 'low_price': 'low', 'volume': 'volume'})
    df['coin'] = coin
    df['date'] = df['date'].astype(str)
    logging.info('casting df data to floats...')
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    buy_signal = 0
    sell_signal = 0
    sell_strength = 0
    hold_signal = 0
    highest_price = df['close'].iloc[0]
    purchase_price = df['close'].iloc[0]
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
    if rsi < 30 and macd > 0:
        buy_signal += 1
    elif rsi > 70 and macd < 0:
        sell_signal += 1
    elif macd > macd_signal and current_price > ma200:
        buy_signal += 1
    elif macd < macd_signal and current_price < ma200:
        sell_signal += 1
    elif current_price < lower_bollingerband:
        buy_signal += 1
    elif current_price > upper_bollingerband:
        sell_signal += 1
    elif current_price > ma200 and ma50 > ma200:
        buy_signal += 1
    elif current_price < ma200 and ma50 < ma200:
        sell_signal += 1
    elif current_price > ma200 and ma50 < ma200:
        buy_signal += 1
    elif current_price < ma200 and ma50 > ma200:
        sell_signal += 1
    else:
        hold_signal += 1
    print(f' --> {coin} (+): {buy_signal} | (-): {sell_signal} | (!): {hold_signal}')
    ticking_iterator += 1
    if ticking_iterator % 10 == 0:
        values_positions = []
        for coin in crypto_I_own:
            current_price = df['close'].iloc[-1]
            values_positions.append(current_price * float(crypto_I_own[coin]))
        median_portfolio = statistics.median(values_positions)
        value_position = current_price * crypto_I_own[coin]
        if value_position > median_portfolio:
            ic()
            sell_signal += 1
    if buy_signal > sell_signal and buy_signal > hold_signal:
        buy_signal = 1
        sell_signal = 0
        hold_signal = 0
    elif sell_signal > buy_signal and sell_signal > hold_signal:
        buy_signal = 0
        sell_signal = 1
        hold_signal = 0
    elif hold_signal > buy_signal and hold_signal > sell_signal:
        buy_signal = 0
        sell_signal = 0
        hold_signal = 1
    for coin in crypto_I_own:
        TOTAL_CRYPTO_DOLLARS += crypto_I_own[coin] * current_price
        break
    if TOTAL_CRYPTO_DOLLARS > TOTAL_CRYPTO_DOLLARS * threshold_total_crypto_per_coin:
        if buy_signal == 1:
            buy_signal = 0
            hold_signal = 1
            sell_signal -= 1
        elif sell_signal == 1:
            sell_signal = 1
            hold_signal = 0
            buy_signal = 0
    return buy_signal, sell_signal, hold_signal
def trading_function():
    """
    The trading_function function is the main function that executes orders based on signals.
    It takes in a dictionary of coin symbols and their corresponding buy, sell, or hold signal.
    The trading_function function then iterates through each coin symbol in the dictionary and
    executes an order if it meets certain criteria:
    :return: A dictionary of the form:
    :doc-author: Trelent
    """
    global signals_dict
    global crypto_I_own
    global loop_count
    global BUYING_POWER
    BUYING_POWER = float(r.profiles.load_account_profile(info='buying_power'))
    time.sleep(20)
    print(f'crypto_I_own: {crypto_I_own}')
    for coin in signals_dict.keys():
        buy_signal = int(signals_dict[coin][0])
        sell_signal = int(signals_dict[coin][1])
        hold_signal = int(signals_dict[coin][2])
        position = r.crypto.get_crypto_positions(info='quantity')
        position = float(position['quantity_available']) if type(position) == dict else 0
        print(f'position: {position}')
        if sell_signal > 0 and position > 0:
            ic()
            try:
                order_crypto(symbol=coin,
                             quantity_or_price=position,
                             amount_in='amount',
                             side='sell',
                             bp=BUYING_POWER,
                             timeInForce='gtc')
            except Exception as e:
                logging.info(f'Unable to generate orders for {coin}...{e}')
        if buy_signal > sell_signal and buy_signal > hold_signal and position == 0 and BUYING_POWER > 0:
            ic()
            order_value = 0.01 * BUYING_POWER
            order_crypto(symbol=coin,
                         quantity_or_price=BUYING_POWER+ 0.25 * buy_signal,
                         amount_in='dollars',
                         side='buy',
                         bp=BUYING_POWER,
                         timeInForce='gtc')
            BUYING_POWER -= order_value
            print(f'BUYING_POWER: {BUYING_POWER}')
            print(f'I just bought {order_value} of {coin}... for ${order_value}...')
        elif sell_signal > buy_signal and sell_signal > hold_signal:
            ic()
            order_crypto(symbol=coin,
                         quantity_or_price= float(position),
                         amount_in='amount',
                         side='sell',
                         bp=BUYING_POWER,
                         timeInForce='gtc')
        else:
            logging.info(f'No action taken for {coin}...')
        time.sleep(random.randint(1, 5))
    logging.info(f'Finished executing orders for {datetime.now(timezone("US/Central"))}...')
def is_daytime():
    current_time = datetime.now(timezone('US/Central'))
    current_hour = current_time.hour
    if current_hour >= 8 and current_hour <= 20:
        return True
    else:
        return False
@sleep_and_retry
async def get_total_crypto_dollars():
    """
    The get_total_crypto_dollars function gets the total value of all crypto owned.
    :return: The total value of all the coins i own
    :doc-author: Trelent
    """
    global TOTAL_CRYPTO_DOLLARS
    global crypto_I_own
    global loop_count
    global BUYING_POWER
    ic()
    print(Fore.BLUE + 'Updating the portfolio value...' + Fore.RESET)
    for coin in tqdm(crypto_I_own.keys()):
        print(Back.YELLOW + Fore.BLACK + f'Getting the current price of {coin}...' + Fore.RESET + Back.RESET, end = '')
        try:
            current_price = float(r.crypto.get_crypto_quote(coin, info='mark_price'))
        except:
            TOTAL_CRYPTO_DOLLARS += 0.0
            print(Fore.RED + f'Unable to get the current price of {coin}...' + Fore.RESET)
            continue
        total_value = crypto_I_own[coin] * current_price
        TOTAL_CRYPTO_DOLLARS += total_value
        print(Fore.GREEN + f'${total_value} of {coin}...\n\t total: ${TOTAL_CRYPTO_DOLLARS}' + Fore.RESET)
    if is_daytime():
        print(Fore.BLUE + 'Sleeping for 3 minutes...' + Fore.RESET)
        await asyncio.sleep(170)
    else:
        print(Fore.BLUE + 'Sleeping for 5 minutes...' + Fore.RESET)
        await asyncio.sleep(300)
async def main():
    global loop_count
    global starting_equity
    global BUYING_POWER
    global start_date
    global TOTAL_CRYPTO_DOLLARS
    global crypto_I_own
    global signals_dict
    global loop_count
    global starting_equity
    starting_equity = BUYING_POWER
    start_date = datetime.now(timezone('US/Central'))
    while True:
        ascii_art_startup = """
            🌲       🌲
        🌲   Sherwood    🌲
            🌲       🌲
        """
        print(Fore.GREEN + ascii_art_startup + Fore.RESET)
        r.orders.cancel_all_crypto_orders()
        print(Fore.YELLOW + 'Cancelling all outstanding orders...' + Fore.RESET)
        print(Fore.YELLOW + 'sleeping for 30 seconds...' + Fore.RESET)
        time.sleep(30)
        await asyncio.to_thread(calculate_ta_indicators)
        await asyncio.to_thread(trading_function)
        if loop_count % 20 == 0:
            print(f'BUYING_POWER: ${BUYING_POWER}')
            print(f'Profit: {BUYING_POWER - starting_equity}')
            print(f'Profit %: {((BUYING_POWER - starting_equity) / starting_equity) * 100:.2f}')
            print(f'Loop count: {loop_count}')
            print(f'Running for {datetime.now(timezone("US/Central")) - start_date}')
        if loop_count % 10 == 0:
            print(Fore.BLUE + 'selling half of every position...' + Fore.RESET)
            for coin in crypto_I_own:
                historical_data = r.crypto.get_crypto_historicals(symbol=coin, interval='5minute', span='hour')
                prices = [float(data_point['close_price']) for data_point in historical_data]
                current_price = prices[-1]
                price_10_minutes_ago = prices[-3]
                price_15_minutes_ago = prices[-4]
                if price_15_minutes_ago > price_10_minutes_ago > current_price:
                    position = float(crypto_I_own[coin])
                    if position > 0:
                        try:
                            sell_quantity = position / 2
                            order_crypto(symbol=coin,
                                        quantity_or_price=sell_quantity,
                                        amount_in='amount',
                                        side='sell',
                                        bp=BUYING_POWER,
                                        timeInForce='gtc')
                        except Exception as e:
                            print(Fore.RED + f' [*] Unable to sell {sell_quantity} of {coin}...' + Fore.RESET)
                            print(e)
                            continue
                        BUYING_POWER += sell_quantity
                        time.sleep(0.25)
                        print(Fore.GREEN + f'I checked historical data and decided to sell {sell_quantity} of {coin}...' + Fore.RESET)
                        crypto_I_own[coin] = position - sell_quantity
        if is_daytime():
            print('daytime mode')
            if BUYING_POWER - starting_equity > 0:
                print(Fore.GREEN + f'Profit % since start of day: {((BUYING_POWER - starting_equity) / starting_equity) * 100:.2f}' + Fore.RESET)
            else:
                print(Fore.RED + f'Profit % since start of day: $({((BUYING_POWER - starting_equity) / starting_equity) * 100:.2f})' + Fore.RESET)
            print(f'Profit % since start of day: {((BUYING_POWER - starting_equity) / starting_equity) * 100:.2f}')
            print('Sleeping for 5 minutes...')
            for i in tqdm(range(300)):
                await asyncio.sleep(1)
        else:
            print('Sleeping for 10 minutes...')
            for i in tqdm(range(600)):
                await asyncio.sleep(1)
@sleep_and_retry
async def update_buying_power():
    ic()
    while True:
        account_details_df = pd.DataFrame(await asyncio.to_thread(r.profiles.load_account_profile, info=None), index=[0])
        BUYING_POWER = float(account_details_df['onbp'])
        if BUYING_POWER > 1.00:
            BUYING_POWER = BUYING_POWER * PERCENTAGE_IN_PLAY
        else:
            BUYING_POWER = BUYING_POWER
        print(Fore.BLUE + f'BUYING_POWER: ${BUYING_POWER}' + Style.RESET_ALL)
        await asyncio.sleep(180)
async def run_async_functions(loop_count, BUYING_POWER):
    ic()
    loop_count += 1
    await asyncio.gather(main(),
                         update_buying_power(),
                         get_total_crypto_dollars(),
                         log_file_size_checker())
def main_looper():
    while True:
        ic()
        loop_count = 0
        start_date = datetime.now(timezone('US/Central'))
        BUYING_POWER = 0
        starting_equity = BUYING_POWER
        print(F'='*50)
        print(Fore.GREEN + 'Buying Power is: {}'.format(BUYING_POWER) + Style.RESET_ALL)
        print(Fore.GREEN + 'Total Profit is: ${}'.format(
            BUYING_POWER - starting_equity) + Style.RESET_ALL)
        try:
            asyncio.run(run_async_functions(loop_count, BUYING_POWER))
        except Exception as e:
            raise e
        while True:
            if loop_count % 20 == 0:
                print(f'BUYING_POWER: {BUYING_POWER}')
                print(f'Profit: {BUYING_POWER - starting_equity}')
                print(f'Profit %: {((BUYING_POWER - starting_equity) / starting_equity) * 100:.2f}')
                print(f'Loop count: {loop_count}')
                print(f'Running for {datetime.now(timezone("US/Central")) - start_date}')
            print('Sleeping for 5 minutes...')
            for i in tqdm(range(300)):
                time.sleep(1)
print('Starting main looper function...')
login_setup()
if RESET:
    areyousure = print(Fore.RED + 'warning destructive action, are you sure? Will commence in 10 seconds...' + Style.RESET_ALL)
    time.sleep(10)
    resetter()
    time.sleep(120)
try:
    print(f'Logging in to Robinhood... and beginning main looper function...')
    main_looper()
except Exception as e:
    time.sleep(3600)
    logging.error(e)
