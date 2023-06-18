from finta import TA
from colorama import Fore
from datetime import datetime
import pandas as pd
import logging, traceback, json, os
from robin_stocks import robinhood as r
from icecream import ic
from pytz import timezone
from tqdm import tqdm
import random
import os
from ratelimit import sleep_and_retry
import ast
import pandas as pd
import asyncio
import time
signals_dict = {}
minimum_orders_coins = {}

#^global variables
verboseMode = True # set to True to see more output
percentage_in_play = 0.60 # 60% of buying power will be used
BUYING_POWER = 0 # initialize buying power
RESET = False # set to True to reset all positions and orders
#^global variables end


async def order_crypto(symbol, quantity_or_price, amount_in='dollars', side='buy', bp=None, timeInForce='gtc'):
    global BUYING_POWER

    if symbol is None:
        return

    symbol = symbol.upper() if type(symbol) == str else str(symbol).upper()
    side = side.lower() if type(side) == str else str(side).lower()
    amount_in = amount_in.lower() if side == 'sell' else 'dollars'
    timeInForce = timeInForce.lower()

    if side == 'buy':
        profile = await asyncio.to_thread(r.profiles.load_account_profile)
        os.environ['PROFILE'] = str(profile)

        if bp is None:
            bp = BUYING_POWER * percentage_in_play

        if symbol == 'DOG':
            symbol = 'DOGE'

        quantity_or_price = max(1.10, 0.10 * float(BUYING_POWER))

        if quantity_or_price > BUYING_POWER:
            if verboseMode:
                print(Fore.RED + f'Not enough buying power to buy {quantity_or_price} {symbol}...' + Fore.RESET)
            return

        await asyncio.sleep(random.randint(1, 3))

    try:
        await asyncio.to_thread(r.orders.order_crypto, symbol=str(symbol), quantityOrPrice=quantity_or_price, amountIn=amount_in, side=side, timeInForce='gtc', jsonify=True)
        await asyncio.sleep(random.randint(1, 3))
    except Exception as e:
        print(Fore.RED + f'Failed try statement in order! {side} because \n{e}' + Fore.RESET)

    if side == 'sell':
        if verboseMode:
            current_price = float((await asyncio.to_thread(r.crypto.get_crypto_quote, symbol))['mark_price'])
            print(Fore.RED + f'Sell order for {quantity_or_price} {symbol} worth ${quantity_or_price * current_price}' + Fore.RESET)
        pass
    else:
        if verboseMode:
            print(Fore.GREEN + f'Buy order at ${quantity_or_price} of {symbol}' + Fore.RESET)
        pass

    return

async def login_setup():
    """
    The login_setup function is used to log into the Robinhood API and return a dataframe of account details.
    It also sets up logging for debugging purposes.
    :return: A dataframe and a login object
    :doc-author: Trelent
    """
    global BUYING_POWER

    with open('config/credentials.json') as f:
        credentials = json.load(f)

    login = await asyncio.to_thread(r.login, credentials['username'], credentials['password'])
    account_details = await asyncio.to_thread(r.profiles.load_account_profile, info=None)

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
async def robin_getter(coin):
    crypto_available_on_robinhood = await asyncio.to_thread(r.crypto.get_crypto_currency_pairs)
    crypto_historicals = await asyncio.to_thread(r.crypto.get_crypto_historicals, str(coin), "5minute", "day", "24_7", info=None)
    crypto_price = await asyncio.to_thread(r.crypto.get_crypto_quote, str(coin))
    return crypto_available_on_robinhood, crypto_historicals, crypto_price

@sleep_and_retry
async def get_crypto_positions_in_account():
    crypto_positions = await asyncio.to_thread(r.crypto.get_crypto_positions)
    positions_dict = {}

    for position in crypto_positions:
        symbol = position['currency']['code']
        quantity = float(position['quantity_available'])
        positions_dict[symbol] = {
            'quantity': quantity
        }

    return positions_dict

async def resetter():
    if RESET:
        crypto_positions = await asyncio.to_thread(r.crypto.get_crypto_positions)

        for position in crypto_positions:
            symbol = position['currency']['code']
            quantity = float(position['quantity_available'])
            order_crypto(symbol, quantity, side='sell')

        orders = await asyncio.to_thread(r.orders.get_all_open_crypto_orders)

        for order in orders:
            await asyncio.to_thread(r.orders.cancel_crypto_order, order['id'])

@sleep_and_retry
async def get_account():
    account = await asyncio.to_thread(r.profiles.load_account_profile, info=None)
    return account

async def brain_module():
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

    coins_list = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']
    os.environ['COINS_LIST'] = str(coins_list)

    minimum_orders_coins = pd.DataFrame()
    holdings_df = pd.DataFrame()

    crypto_positions_dict = await get_crypto_positions_in_account()
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

            coin_info = await asyncio.to_thread(r.crypto.get_crypto_info, coin)
            coin_info_df = pd.DataFrame(coin_info, index=[index_of_coin])
            minimum_orders_coins = minimum_orders_coins.append(coin_info_df)
            index_of_coin += 1

        minimum_orders_coins = minimum_orders_coins.set_index('symbol')
        minimum_orders_coins.to_csv('data/minimum_orders_coins.csv')

    coin_historicals_dfs = []

    for coin in tqdm(coins_list):
        crypto_available_on_robinhood, crypto_historicals, crypto_price = await robin_getter(coin)
        coin_historicals_dfs.append(crypto_historicals)
        crypto_data = {
            "coin": coin,
            "crypto_available_on_robinhood": crypto_available_on_robinhood,
            "crypto_historicals": crypto_historicals,
            "coin_mark_price": float(crypto_price['mark_price']),
            "coin_ask_price": float(crypto_price['ask_price']),
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
        buy_signal, sell_signal, hold_signal = await signal_engine(df, coin)
        signals_dict[coin] = {
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "hold_signal": hold_signal
        }

    with open('data/signals_dict.json', 'w') as fp:
        signals_json = json.dumps(signals_dict)
        fp.write(signals_json)

    return signals_dict

async def signal_engine(df, coin):
    """
    The signal_engine function takes a dataframe and a coin and returns buy, sell, and hold signals.
    :param df: The dataframe
    :param coin: The coin
    :return: The buy, sell, and hold signals
    :doc-author: Trelent
    """
    global minimum_orders_coins

    if coin not in minimum_orders_coins.index:
        return False, False, False

    minimum_order_price = minimum_orders_coins.loc[coin, 'min_order_price']

    if df is None or df.empty:
        return False, False, False

    try:
        df = df.sort_values(by='begins_at', ascending=False)
        last_close_price = float(df.iloc[0]['close_price'])
        second_last_close_price = float(df.iloc[1]['close_price'])
        difference_in_close_price = abs(last_close_price - second_last_close_price)

        if difference_in_close_price / last_close_price > 0.03:
            return False, False, False

        df['sma_5'] = TA.SMA(df, 5)
        df['sma_10'] = TA.SMA(df, 10)
        df['sma_20'] = TA.SMA(df, 20)
        df['sma_50'] = TA.SMA(df, 50)
        df['sma_200'] = TA.SMA(df, 200)

        df['ema_5'] = TA.EMA(df, 5)
        df['ema_10'] = TA.EMA(df, 10)
        df['ema_20'] = TA.EMA(df, 20)
        df['ema_50'] = TA.EMA(df, 50)
        df['ema_200'] = TA.EMA(df, 200)

        df['macd'] = TA.MACD(df)['MACD']
        df['macd_signal'] = TA.MACD(df)['SIGNAL']

        df['rsi_14'] = TA.RSI(df, 14)
        df['rsi_7'] = TA.RSI(df, 7)

        df['stoch'] = TA.STOCH(df)['STOCH']
        df['stoch_signal'] = TA.STOCH(df)['STOCH_SIGNAL']

        df = df.iloc[::-1]

        if df['sma_5'].iloc[0] > df['sma_10'].iloc[0] > df['sma_20'].iloc[0] > df['sma_50'].iloc[0] > df['sma_200'].iloc[0]:
            if df['macd'].iloc[0] > df['macd_signal'].iloc[0] and df['macd'].iloc[0] < 0:
                if df['rsi_14'].iloc[0] > 30 and df['rsi_14'].iloc[0] < 70:
                    if df['stoch'].iloc[0] > df['stoch_signal'].iloc[0] and df['stoch'].iloc[0] < 20:
                        if df['close_price'].iloc[0] > df['sma_200'].iloc[0]:
                            if last_close_price <= minimum_order_price:
                                return False, False, False
                            else:
                                return True, False, False

        if df['sma_5'].iloc[0] < df['sma_10'].iloc[0] < df['sma_20'].iloc[0] < df['sma_50'].iloc[0] < df['sma_200'].iloc[0]:
            if df['macd'].iloc[0] < df['macd_signal'].iloc[0] and df['macd'].iloc[0] > 0:
                if df['rsi_14'].iloc[0] > 30 and df['rsi_14'].iloc[0] < 70:
                    if df['stoch'].iloc[0] < df['stoch_signal'].iloc[0] and df['stoch'].iloc[0] > 80:
                        return False, True, False

        return False, False, True

    except Exception as e:
        logging.warning('Exception in signal_engine: {}'.format(str(e)))
        return False, False, False

async def main():
    global BUYING_POWER

    account_details_df = await get_account()

    # convert to df if not already
    if not isinstance(account_details_df, pd.DataFrame):
        account_details_df = pd.DataFrame(account_details_df)
    if account_details_df is None or account_details_df.empty:
        print(Fore.RED + 'Failed to load account details. Please check your credentials and try again.' + Fore.RESET)
        return

    BUYING_POWER = float(new_func(account_details_df))
    print(Fore.GREEN + 'Buying Power: $' + '{:,.2f}'.format(BUYING_POWER) + Fore.RESET)

    signals_dict = await brain_module()

    for coin, signals in signals_dict.items():
        if signals['buy_signal']:
            quantity_to_buy = BUYING_POWER / signals['coin_mark_price']
            quantity_to_buy = math.floor(quantity_to_buy * 100000000) / 100000000
            order_crypto(coin, quantity_to_buy, side='buy')

        if signals['sell_signal']:
            quantity_to_sell = crypto_I_own[coin]
            order_crypto(coin, quantity_to_sell, side='sell')

        if signals['hold_signal']:
            continue

    print(Fore.GREEN + 'Execution complete.' + Fore.RESET)
    return

def new_func(account_details_df):
    return account_details_df['onbp'][-1]

if __name__ == '__main__':
    # let's log in to Robinhood
    with open('config/credentials.json') as json_file:
        credentials = json.load(json_file)
        r.login(username=credentials['username'], password=credentials['password'])
    print(Fore.GREEN + 'Successfully logged in to Robinhood!' + Fore.RESET)
    asyncio.run(main()) # Python 3.7+
    # run the buying/selling loop
    while True:
        time.sleep(60)
        asyncio.run(main())
