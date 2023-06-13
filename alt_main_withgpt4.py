from finta import TA
from colorama import Fore, Back, Style
from datetime import datetime, timedelta
import matplotlib.pyplot as plt, pandas as pd, numpy as np
import logging, traceback, json, time, csv, os
from robin_stocks import robinhood as r
# import icecream
from icecream import ic
from pytz import timezone
from tqdm import tqdm
import random
import os
from time import sleep
from ratelimit import limits, sleep_and_retry
signals_dict = {}
minimum_orders_coins = {}
import ast
import re
import pandas as pd




@sleep_and_retry
def order_crypto(symbol, quantity_or_price, amount_in='dollars', side='buy', bp=None, timeInForce='gtc'):
    #ic()
    global BUYING_POWER
    if symbol is None:
        return
    symbol = symbol.upper() if type(symbol) == str else str(symbol).upper()
    side = side.lower() if type(side) == str else str(side).lower()
    amount_in = amount_in.lower() if side == 'sell' else 'dollars'
    timeInForce = timeInForce.lower()

    #print(Fore.GREEN + f'{side} {quantity_or_price} {symbol}...' + Fore.RESET)
    if side == 'buy':
        profile = r.profiles.load_account_profile()
        os.environ['PROFILE'] = str(profile) # set an env variable for the profile
        if bp is None:
            bp = BUYING_POWER
        # bp = float(bp)
        if symbol == 'DOG': symbol = 'DOGE' #todo hacked in
        quantity_or_price = max(quantity_or_price, BUYING_POWER * 0.01, minimum_orders_coins.get(symbol, float('inf')))
        time.sleep(random.randint(1, 3))
        #print(f'Set quantity_or_price to {quantity_or_price} {symbol}...')
        #print(f'Quantity_or_price is {quantity_or_price} {symbol}...')
        #print(f'Buying power is {bp}...')
        #print(f'Minimum order amount is {minimum_orders_coins.get(symbol, float("inf"))} {symbol}...')
    try:
        #print(f'Attempting a {side} order of {quantity_or_price} {symbol}...')
        r.orders.order_crypto(symbol=str(symbol), quantityOrPrice=quantity_or_price, amountIn=amount_in, side=side, timeInForce='gtc', jsonify=True)
        time.sleep(random.randint(1, 3))
        #print(f'{side.capitalize()}ing {quantity_or_price} {symbol}...')
        #print(Fore.GREEN + f'Passed try statement in order! {side}' + Fore.RESET)
        time.sleep(1)
    except Exception as e:
        print(Fore.RED + f'Failed try statement in order! {side} because \n{e}' + Fore.RESET)

    print(Fore.GREEN + f'order for a {side} of {symbol} at ${quantity_or_price} {amount_in} was successful!' + Fore.RESET)
def login_setup():
    #ic()
    global BUYING_POWER

    with open('config/credentials.json') as f:
        credentials = json.load(f)
    login = r.login(credentials['username'], credentials['password'])
    account_details = r.profiles.load_account_profile(info=None)
    if isinstance(account_details, dict):
        account_details_df = pd.DataFrame(account_details, index=[0])
    else:
        # convert to dict then to df
        print(Fore.BLUE + "DEBUGGING -> SETTING account_details to a dictionary" + Fore.RESET)
        account_details_df = pd.DataFrame(account_details.to_dict(), index=[0])

    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(filename='logs/robinhood.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.info('Started')
    BUYING_POWER = float(account_details_df['onbp']) #! this is a global variable
    return account_details_df, login
def robin_getter(coin):
    #ic()
    crypto_available_on_robinhood = r.crypto.get_crypto_currency_pairs()
    crypto_historicals = r.crypto.get_crypto_historicals(str(coin), "5minute", "day")
    crypto_price = r.crypto.get_crypto_quote(str(coin))
    return crypto_available_on_robinhood, crypto_historicals, crypto_price
@sleep_and_retry
def get_crypto_positions_in_account():
    #ic()

    crypto_positions = r.crypto.get_crypto_positions()
    positions_dict = {}
    for position in crypto_positions:
        symbol = position['currency']['code']
        quantity = float(position['quantity_available'])
        positions_dict[symbol] = {
            'quantity': quantity
        }
    return positions_dict
def brain_module():
    #ic()
    """
    The brain_module function is the main function of this program. It does the following:
        1) Gets a list of coins to trade from an env variable called COINS_LIST
        2) Gets a dictionary of crypto I own from an env variable called CRYPTO_I_OWN
        3) Creates a dataframe with columns for each coin in COINS_LIST and rows for each coin in CRYPTO_I OWN,
            where the values are how much crypto I own (in dollars). This dataframe is saved as holdings.csv.

    :return: A dictionary of the signals for each coin
    :doc-author: Trelent
    """
    global crypto_I_own
    global holdings_df
    global minimum_orders_coins
    global crypto_positions_df
    global BUYING_POWER
    coins_list = ['BTC', 'ETH', 'ADA', 'DOGE', 'MATIC', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'SOL', 'AVAX', 'XLM', 'BCH', 'XTZ']
    # set an env variable for the coins list
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
    #print(f'Getting the minimum order amount for each coin...')
    # check if the minimum csv file already exists
    if os.path.exists('data/minimum_orders_coins.csv'):
        #ic()
        minimum_orders_coins = pd.read_csv('data/minimum_orders_coins.csv')
        minimum_orders_coins = minimum_orders_coins.set_index('symbol')
    else:
        ic()
        index_of_coin = 0
        for coin in coins_list:
            ic()
            #tqdm.write(f'Getting the minimum order amount for {coin}...')
            coin_info = r.crypto.get_crypto_info(coin)
            coin_info_df = pd.DataFrame(coin_info, index=[index_of_coin])
            minimum_orders_coins = minimum_orders_coins.append(coin_info_df)
            # save the minimum order amount for each coin


            index_of_coin += 1
        minimum_orders_coins = minimum_orders_coins.set_index('symbol')
        minimum_orders_coins.to_csv('data/minimum_orders_coins.csv')
    coin_historicals_dfs = []




    for coin in coins_list:
        #ic()
        tqdm.write(f'Getting the historical data for {coin}...')
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
    #print(f'Saving data... to data/ as json files...')
    if not os.path.exists('data'):
        #ic()
        os.makedirs('data')
    with open('data/crypto_I_own.json', 'w') as fp:
        ic()
        json.dump(crypto_I_own, fp)
    with open('data/holdings_df.json', 'w') as fp:
        holdings_dict = holdings_df.to_dict()
        holdings_json = json.dumps(holdings_dict)
        fp.write(holdings_json)
    # with open('data/minimum_orders_coins.json', 'w') as fp:
    #     minimum_orders_coins_dict = minimum_orders_coins.to_dict(minimum_orders_coins)
    #     minimum_orders_coins_json = json.dumps(minimum_orders_coins_dict)
    #     fp.write(minimum_orders_coins_json)
    #print(f'Calculating signals...')
    logging.info('Calculating signals...')
    signals_dict = {}
    for df in coin_historicals_dfs:
        # ic()
        crypto_historicals = df
        crypto_historicals_df = pd.DataFrame(crypto_historicals)
        if 'USD' in str(crypto_historicals_df['symbol'][0]):
            ic()
            coin = str(crypto_historicals_df['symbol'][0])[:3]
        elif '-USD' in str(crypto_historicals_df['symbol'][0]):
            ic()
            coin = str(crypto_historicals_df['symbol'][0])[:4]
        elif 'DOGE' in str(crypto_historicals_df['symbol'][0]):
            ic()
            coin = str("DOGE")
        else:
            ic()
            coin = str(crypto_historicals_df['symbol'][0])
        logging.info('  Calculating signals for {}...'.format(coin))
        buy_signal, sell_signal, hold_signal = signal_engine(df, coin)
        signals_dict[coin] = [buy_signal, sell_signal, hold_signal]
    for coin in coins_list:
        # ic()
        if coin not in signals_dict.keys():
            # add it with 0 signals
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
            #^ signal is that we should buy
            if buy_signal > sell_signal and buy_signal > hold_signal:
                cryptoiownlist = crypto_I_own
                # use env vars to use order_crypto
                if type(crypto_I_own) == dict and coin in crypto_I_own.keys():
                    # get avg number of holdings for all other coins in the portfolio
                    # avg_holdings [float(crypto_I_own[coin]) for coin in crypto_I_own.keys() if coin != coin and float(crypto_I_own[coin]) > 0.01]
                    # avg_holdings = sum(avg_holdings) / len(avg_holdings)
                    # if float(crypto_I_own[coin]) > avg_holdings:
                    #     # we own this coin, and we have more than the avg holdings
                    #     ic()
                    #     pass
                    # else:
                    #     # we own this coin, but we have less than the avg holdings so we can buy more
                    #     order_crypto(symbol=str(coin),  # coin
                    #             quantity_or_price=max(1.00,0.10 * float(BUYING_POWER)),  # 5% of buying power
                    #             amount_in='dollars',  # dollars
                    #             side='buy',  # buy
                    #             timeInForce='gtc')
                    #     continue
                    pass
                elif type(crypto_I_own) == str and coin in crypto_I_own:
                    if float(crypto_I_own[coin]) > 0.01:
                        #print(f'Already own {coin}...')
                        #ic()
                        pass
                else:
                    print(type(crypto_I_own))
                    print(crypto_I_own)
                    #print(f'Buying {coin}...')
                    ic()
                order_crypto(symbol=str(coin),  # coin
                                quantity_or_price=max(1.00,0.10 * float(BUYING_POWER)),  # 5% of buying power
                                amount_in='dollars',  # dollars
                                side='buy',  # buy
                                timeInForce='gtc')  # good till cancel
            elif sell_signal > buy_signal and sell_signal > hold_signal:
                coins_I_own = crypto_I_own
                # convert to dict with ast
                if type(coins_I_own) == str:
                    coins_I_own = ast.literal_eval(coins_I_own)
                else:# the type is already a dict
                    pass
                # if we don't own any of this coin, or we own less than 0.01 of it, skip it
                coins_owned = coins_I_own[coin]
                # convert to float
                coins_owned = float(coins_owned)
                if coins_owned < 0.01 or coin not in coins_I_own.keys():
                    #print(f'Not enough {coin} to sell...')
                    continue
                print(f'Selling {coin}...')
                order_crypto(symbol=str(coin),
                                quantity_or_price=0.80 * float(crypto_I_own[coin]),
                                amount_in='amount',
                                side='sell',
                                timeInForce='gtc')

                #print(f'Selling {coin}...')
                time.sleep(1)
            elif hold_signal > buy_signal and hold_signal > sell_signal:
                #print(f'Hold {coin}...')
                ic()
            elif buy_signal == sell_signal and buy_signal == hold_signal:
                #print(f'Hold {coin}... as buy_signal == sell_signal == hold_signal')
                ic()
            elif buy_signal == sell_signal:
                #print(f'Hold {coin}... as buy_signal == sell_signal')
                ic()
            else:
                ic()
                #print(f'Hold {coin}... \n buy_signal: {buy_signal} \n sell_signal: {sell_signal} \n hold_signal: {hold_signal}')
        except Exception as e:
            traceback.print_exc()
            logging.info(' {} is the error'.format(e))
            #print(f'Hold {coin}... \n buy_signal: {buy_signal} \n sell_signal: {sell_signal} \n hold_signal: {hold_signal}')
            logging.info('  {} buy signal: {}'.format(coin, buy_signal))
            logging.info('  {} sell signal: {}'.format(coin, sell_signal))
            logging.info('  {} hold signal: {}'.format(coin, hold_signal))
            continue
    # save signals dict as a env var
    os.environ['CRYPTO_SIGNALS'] = str(signals_dict)
    # SET THE GLOBAL VARIABLE `crypto_signals` TO THE `signals_dict` VARIABLE
    global crypto_signals
    crypto_signals = signals_dict



def signal_engine(df, coin):
    #ic()
    global signals_dict
    global crypto_I_own
    df = pd.DataFrame(df)
    coin = str(coin)
    df = df[['begins_at', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']]
    df = df.rename(columns={'begins_at': 'date', 'open_price': 'open', 'close_price': 'close', 'high_price': 'high', 'low_price': 'low', 'volume': 'volume'})
    df['coin'] = coin
    df['date'] = df['date'].astype(str)
    logging.info('  df date: {}'.format(df['date']))
    logging.info('casting df data to floats...')
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    buy_signal = 0
    sell_signal = 0
    hold_signal = 0
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
    # logging.info('  {} buy signal: {}'.format(coin, buy_signal))
    # logging.info('  {} sell signal: {}'.format(coin, sell_signal))
    # logging.info('  {} hold signal: {}'.format(coin, hold_signal))
    return buy_signal, sell_signal, hold_signal

def action_engine():
    """
    The action_engine function is the main function that executes all of the other functions.
    It takes no arguments, but it does require a signals_dict variable to be defined in order to work properly.
    The action_engine function will iterate through each coin in the signals_dict and execute an order based on
    the signal values for that coin.

    :return: A dictionary of signals for each coin
    :doc-author: Trelent
    """
    global signals_dict
    global crypto_I_own
    global loop_count
    global BUYING_POWER
    BUYING_POWER = float(r.profiles.load_account_profile(info='buying_power'))

    #signals_dict = os.environ['CRYPTO_SIGNALS'].replace('\n', '')  # this is a string

    # force signals_dict to be 1, 0, or -1 for buy, hold, sell under each coin
    # now make the string a dictionary by converting it to a list
    # signals_dict = signals_dict.split(',')
    # # now take this list and convert it to a dictionary
    # signals_dict = dict(zip(signals_dict[::2], signals_dict[1::2]))
    # # now convert the values in the dictionary to integers
    # signals_dict = {k: int(v) for k, v in signals_dict.items()}
    # signals_dict = ast.literal_eval(signals_dict)  # convert string to dict
    # logging.info(f'  signals_dict: {signals_dict}')
    # print(f'Buying power is ${BUYING_POWER}, proceeding with orders...')
    time.sleep(20)
    print(f'crypto_I_own: {crypto_I_own}')
    for coin in signals_dict.keys():  # iterate through each coin in the signals_dict
        buy_signal = int(signals_dict[coin][0])
        sell_signal = int(signals_dict[coin][1])
        hold_signal = int(signals_dict[coin][2])

        position = r.crypto.get_crypto_positions(info='quantity')
        position = float(position['quantity_available']) if type(position) == dict else 0
        print(f'position: {position}')
        if sell_signal > 0 and position > 0:
            try:
                order_crypto(symbol=coin,
                             quantity_or_price=position,
                             amount_in='amount',
                             side='sell',
                             bp=BUYING_POWER,
                             timeInForce='gtc')
            except Exception as e:
                logging.info(f'Unable to generate orders for {coin}...{e}')
        # signal is a buy signal and we don't already own the coin
        if buy_signal > sell_signal and buy_signal > hold_signal:
            order_value = 0.01 * BUYING_POWER
            order_crypto(symbol=coin,
                         #*quantity_or_price=order_value if order_value > 1.00 else 1.00,
                         quantity_or_price=BUYING_POWER,
                         amount_in='dollars',
                         side='buy',
                         bp=BUYING_POWER,
                         timeInForce='gtc')
            BUYING_POWER -= order_value
            print(f'BUYING_POWER: {BUYING_POWER}')
            print(f'I just bought {order_value} of {coin}... for ${order_value}...')
        # signal is a sell signal and we already own the coin
        elif sell_signal > buy_signal and sell_signal > hold_signal:
            order_crypto(symbol=coin,
                         quantity_or_price= float(r.crypto.get_crypto_positions()[0]['quantity']),
                         amount_in='amount',
                         side='sell',
                         bp=BUYING_POWER,
                         timeInForce='gtc')
        else:
            logging.info(f'No action taken for {coin}...')

        time.sleep(random.randint(1, 5))

    logging.info(f'Finished executing orders for {datetime.now(timezone("US/Central"))}...')

def is_daytime():
    #ic()

    current_time = datetime.now(timezone('US/Central'))
    current_hour = current_time.hour
    if current_hour >= 8 and current_hour <= 20:
        return True
    else:
        return False
if __name__ == '__main__':
    login_setup()
    loop_count = 0
    while True:
        ic()
        try:
            brain_module()
            action_engine()
            if is_daytime():
                print('daytime mode')
                #print('Sleeping for 5 minutes...')
                time.sleep(300)
            else:
                #print('Sleeping for 30 minutes...')
                time.sleep(1800)
            loop_count += 1
        except Exception as e:
            r.orders.cancel_all_crypto_orders()
            logging.info(f'Exception occurred: {e}')
            time.sleep(60)