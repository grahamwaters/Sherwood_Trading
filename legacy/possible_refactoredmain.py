import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from pytz import timezone
from robin_stocks import robinhood as r
from tqdm import tqdm

import pandas as pd

from colorama import Fore, Style
from finta import TA
from ratelimit import limits, sleep_and_retry

# Global variables
signals_dict = {}
minimum_orders_coins = {}
crypto_I_own = {}
BUYING_POWER = 0.0
PERCENTAGE_IN_PLAY = 0.60
percentage_in_play = PERCENTAGE_IN_PLAY
threshold_total_crypto_per_coin = 0.10
verboseMode = True
RESET = True
stop_loss_percent = 0.05
ticking_iterator = 0
TOTAL_CRYPTO_DOLLARS = 0.0

coins_list = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']


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
    """
    global BUYING_POWER

    symbol = str(symbol).upper()
    side = str(side).lower()
    amount_in = str(amount_in).lower() if side == 'sell' else 'dollars'
    timeInForce = str(timeInForce).lower()

    if side == 'buy':
        profile = r.profiles.load_account_profile()
        os.environ['PROFILE'] = str(profile)

        if bp is None:
            bp = BUYING_POWER * percentage_in_play

        if symbol == 'DOG':
            symbol = 'DOGE'

        quantity_or_price = max(1.05, 0.10 * float(BUYING_POWER))

        if quantity_or_price > BUYING_POWER:
            if verboseMode:
                print(Fore.RED + f'Not enough buying power to buy {quantity_or_price} {symbol}...' + Fore.RESET)
            return

        time.sleep(random.randint(1, 3))

    try:
        r.orders.order_crypto(
            symbol=str(symbol),
            quantityOrPrice=quantity_or_price,
            amountIn=amount_in,
            side=side,
            timeInForce='gtc',
            jsonify=True
        )
        time.sleep(random.randint(1, 3))
    except Exception as e:
        print(Fore.RED + f'Failed try statement in order! {side} because \n{e}' + Fore.RESET)

    if side == 'sell':
        if verboseMode:
            current_price = float(r.crypto.get_crypto_quote(symbol)['mark_price'])
            print(Fore.LIGHTMAGENTA_EX + f'Order filled to sell {quantity_or_price} {symbol} at {current_price} '
                                         f'per coin.' + Fore.RESET)
        # Reset our basis of where we got in
        crypto_I_own[symbol]['basis'] = None
        crypto_I_own[symbol]['last_sold_price'] = None
        crypto_I_own[symbol]['percentage_change'] = None
    elif side == 'buy':
        if verboseMode:
            current_price = float(r.crypto.get_crypto_quote(symbol)['mark_price'])
            print(Fore.GREEN + f'Order filled to buy {quantity_or_price} {symbol} at {current_price} '
                               f'per coin.' + Fore.RESET)
        if symbol not in crypto_I_own:
            crypto_I_own[symbol] = {}
        crypto_I_own[symbol]['basis'] = float(quantity_or_price)
        crypto_I_own[symbol]['last_sold_price'] = None
        crypto_I_own[symbol]['percentage_change'] = None
    time.sleep(random.randint(1, 3))
    return


@sleep_and_retry
def login_setup():
    """
    The login_setup function is used to log in to the Robinhood API and set up the necessary environment.
    """
    logging.info('Trying to login...')
    login_success = False
    while not login_success:
        try:
            with open('config.json') as json_file:
                config = json.load(json_file)

            rh_username = config['RH_USERNAME']
            rh_password = config['RH_PASSWORD']
            rh_mfa_code = config['RH_MFA_CODE']
            rh_device_token = config['RH_DEVICE_TOKEN']

            r.login(username=rh_username,
                    password=rh_password,
                    expiresIn=86400,
                    by_sms=True,
                    mfa_code=rh_mfa_code,
                    store_session=True,
                    device_token=rh_device_token)

            login_success = True
        except Exception as e:
            logging.info(f'Failed to login! {e}')
            time.sleep(5)

    time.sleep(random.randint(1, 3))
    return


@sleep_and_retry
def robin_getter(symbol, time_frame='day'):
    """
    The robin_getter function is used to get the historical prices for a given symbol.
    :param symbol: Specify the symbol you want to retrieve data for
    :param time_frame: Specify the time frame for the historical data
    :return: A pandas DataFrame with the historical prices
    """
    global reset_every_minute
    reset_every_minute = 0
    while reset_every_minute == 0:
        reset_every_minute = int(datetime.now(timezone('US/Eastern')).strftime("%M"))
        time.sleep(3)
    try:
        historical_data = r.stocks.get_stock_historicals(symbol, interval='5minute', span=time_frame.upper(),
                                                        bounds='regular')
    except Exception as e:
        logging.error(f'Failed to retrieve historical data! {e}')
        return pd.DataFrame()

    if not historical_data:
        return pd.DataFrame()

    df = pd.DataFrame(historical_data)
    df['begins_at'] = pd.to_datetime(df['begins_at'])
    df.set_index('begins_at', inplace=True)
    df = df.astype(float)
    time.sleep(random.randint(1, 3))
    return df


@sleep_and_retry
def get_crypto_positions_in_account():
    """
    The get_crypto_positions_in_account function is used to retrieve the current positions in the account.
    :return: A dict with the current positions in the account
    """
    positions = r.crypto.get_crypto_positions(info=None)
    if positions is not None and positions != []:
        return {position['currency']['code']: float(position['quantity']) for position in positions}
    else:
        return {}


def compute_signals(crypto_prices, coin):
    """
    The compute_signals function is used to compute trading signals for a given cryptocurrency.
    :param crypto_prices: A DataFrame with historical prices
    :param coin: The cryptocurrency symbol
    :return: A trading signal (buy or sell) for the cryptocurrency
    """
    close_prices = crypto_prices['close']
    stoch_rsi = TA.STOCHRSI(close_prices)

    last_stoch_rsi = stoch_rsi[-1]
    prev_stoch_rsi = stoch_rsi[-2]

    if last_stoch_rsi >= 0.8 and prev_stoch_rsi < 0.8:
        return 'sell'
    elif last_stoch_rsi <= 0.2 and prev_stoch_rsi > 0.2:
        return 'buy'
    else:
        return 'hold'


def process_signal(coin, signal):
    """
    The process_signal function is used to process a trading signal for a given cryptocurrency.
    :param coin: The cryptocurrency symbol
    :param signal: The trading signal (buy, sell, or hold)
    """
    global threshold_total_crypto_per_coin, minimum_orders_coins

    if coin in crypto_I_own:
        if signal == 'sell':
            if verboseMode:
                print(Fore.LIGHTMAGENTA_EX + f"Signal to sell {coin}!" + Fore.RESET)
            quantity = crypto_I_own[coin]['basis']
            if quantity is not None:
                if quantity > threshold_total_crypto_per_coin:
                    order_crypto(coin, quantity, amount_in='shares', side='sell')
                else:
                    if verboseMode:
                        print(Fore.RED + f'Not enough {coin} to sell!' + Fore.RESET)
            else:
                if verboseMode:
                    print(Fore.RED + f'No {coin} in your portfolio!' + Fore.RESET)
        elif signal == 'buy':
            if verboseMode:
                print(Fore.GREEN + f"Signal to buy {coin}!" + Fore.RESET)
            if coin not in minimum_orders_coins:
                minimum_orders_coins[coin] = False
            if minimum_orders_coins[coin] is False:
                order_crypto(coin, 1, amount_in='dollars', side='buy')
            else:
                if verboseMode:
                    print(Fore.RED + f"Already reached minimum order for {coin}!" + Fore.RESET)
        else:
            if verboseMode:
                print(Fore.CYAN + f"Signal to hold {coin}!" + Fore.RESET)
    else:
        if signal == 'buy':
            if verboseMode:
                print(Fore.GREEN + f"Signal to buy {coin}!" + Fore.RESET)
            if coin not in minimum_orders_coins:
                minimum_orders_coins[coin] = False
            if minimum_orders_coins[coin] is False:
                order_crypto(coin, 1, amount_in='dollars', side='buy')
            else:
                if verboseMode:
                    print(Fore.RED + f"Already reached minimum order for {coin}!" + Fore.RESET)
        else:
            if verboseMode:
                print(Fore.CYAN + f"Signal to hold {coin}!" + Fore.RESET)


def main():
    """
    The main function is the entry point of the script.
    """
    global coins_list

    login_setup()
    while True:
        try:
            logging.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("---------------NEW TICK---------------")

            positions = get_crypto_positions_in_account()
            for coin in coins_list:
                if coin in positions:
                    crypto_I_own[coin] = {
                        'basis': positions[coin],
                        'last_sold_price': None,
                        'percentage_change': None
                    }
                else:
                    crypto_I_own[coin] = {
                        'basis': None,
                        'last_sold_price': None,
                        'percentage_change': None
                    }

            for coin in tqdm(coins_list):
                crypto_prices = robin_getter(coin, time_frame='day')
                if not crypto_prices.empty:
                    signal = compute_signals(crypto_prices, coin)
                    process_signal(coin, signal)
                else:
                    logging.error(f'Failed to retrieve data for {coin}.')

            time.sleep(60)
        except KeyboardInterrupt:
            print('Exiting...')
            break


if __name__ == '__main__':
    main()
