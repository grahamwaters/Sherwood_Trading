from robin_stocks import robinhood as r
import pandas as pd
from datetime import datetime
import logging
from sys import exit
import traceback
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
r.login('username', 'password')
panic_mode = False
cryptos = ['BTC', 'ETH', 'ADA', 'DOGE', 'MATIC', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'SOL', 'AVAX', 'XLM', 'BCH', 'XTZ']
cryptos = ['BTC', 'ETH', 'ADA']
crypto_data = pd.DataFrame()
logger = logging.getLogger('crypto_trader')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('crypto_trader.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
from datetime import datetime
from datetime import datetime
import time
now = datetime.now()
buying_limiter = 0.60
crypto_data = pd.DataFrame()
def get_crypto_data(cryptos, crypto_data):
    for crypto in tqdm(cryptos):
        try:
            historicals = r.get_crypto_historicals(crypto, interval='day', span='week')
            current_price = historicals[-1]['close_price']
            historical_prices = [x['close_price'] for x in historicals]
            profile = r.profiles.load_account_profile()
            buying_power = profile['buying_power']
            positions = r.crypto.get_crypto_positions()
            print(f'found {len(positions)} positions.')
            for position in tqdm(positions):
                if position['currency']['code'] == crypto:
                    pos_dict = position['currency']
                    min_order_size = float(pos_dict['increment'])
                    coin_holdings = float(position['quantity_available'])
            profile_data = r.profiles.load_portfolio_profile()
            current_equity = profile_data['equity']
            current_equity = float(current_equity)
            previous_equity = profile_data['adjusted_equity_previous_close']
            previous_equity = float(previous_equity)
            daily_profit = float(current_equity) - float(previous_equity)
            historical_prices = [float(x) for x in historical_prices]
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
            print(f'Error getting data for {crypto}: {e}')
            logger.error(f'Error getting data for {crypto}: {e}')
            traceback.print_exc()
            pass
    def add_me(crypto_data, metric, metric_name):
        if isinstance(metric, pd.Series):
            crypto_data.loc[index, metric_name] = metric.iloc[-1]
        elif isinstance(metric, float):
            crypto_data.loc[index, metric_name] = metric
        elif isinstance(metric, np.ndarray):
            crypto_data.loc[index, metric_name] = metric[-1]
        else:
            crypto_data.loc[index, metric_name] = metric
        return crypto_data
    for index, row in tqdm(crypto_data.iterrows()):
        prices = pd.Series(row['historical_prices'])
        if not isinstance(prices[0], float):
            prices = prices.apply(lambda x: float(x))
        up_df, down_df = prices.copy(), prices.copy()
        up_df[up_df < up_df.shift(1)] = 0
        down_df[down_df > down_df.shift(1)] = 0
        up_df.iloc[0] = 0
        down_df.iloc[0] = 0
        roll_up1 = up_df.ewm(span=14).mean()
        roll_down1 = down_df.ewm(span=14).mean()
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))
        rsi = RSI1
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        exp3 = macd.ewm(span=9, adjust=False).mean()
        macd = macd - exp3
        macd = macd.iloc[-1]
        macd_signal = exp3.iloc[-1]
        macd_hist = macd - macd_signal
        ma200 = prices.rolling(200).mean().iloc[-1]
        ma50 = prices.rolling(50).mean().iloc[-1]
        ma20 = prices.rolling(20).mean().iloc[-1]
        sma = prices.rolling(5).mean().iloc[-1]
        ema = prices.ewm(span=5, adjust=False).mean().iloc[-1]
        window = 14
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        upper_band = pd.Series()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        print(f'Adding technical indicators for {row["ticker"]} to dataframe\n\t...')
        crypto_data = add_me(crypto_data, rsi, 'rsi')
        crypto_data = add_me(crypto_data, macd, 'macd')
        crypto_data = add_me(crypto_data, rolling_mean, 'sma')
        crypto_data = add_me(crypto_data, ema, 'ema')
        crypto_data = add_me(crypto_data, ma200, 'ma200')
        crypto_data = add_me(crypto_data, ma50, 'ma50')
        crypto_data = add_me(crypto_data, ma20, 'ma20')
        crypto_data = add_me(crypto_data, sma, 'sma')
        crypto_data = add_me(crypto_data, ema, 'ema')
        crypto_data = add_me(crypto_data, macd_signal, 'macd_signal')
        crypto_data = add_me(crypto_data, macd_hist, 'macd_hist')
        crypto_data = add_me(crypto_data, upper_band, 'upper_band')
        crypto_data = add_me(crypto_data, lower_band, 'lower_band')
    crypto_data.to_csv('crypto_data.csv', index=False)
    return crypto_data
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
        return [x]
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
    crypto_data[crypto_data.columns[1:]] = crypto_data[crypto_data.columns[1:]].apply(pd.to_numeric, errors='coerce')
    crypto_data['pct_change_5h'] = crypto_data.apply(lambda x: calculate_pct_change_5h(x['historical_prices'], x['current_price']), axis=1)
    crypto_data['signal'] = 0
    crypto_data.loc[(crypto_data['rsi'] < 30) & (crypto_data['macd'] > 0), 'signal'] += 1
    crypto_data.loc[(crypto_data['rsi'] > 70) & (crypto_data['macd'] < 0), 'signal'] -= 1
    crypto_data.loc[(crypto_data['macd'] > crypto_data['macd_signal']) & (crypto_data['current_price'] > crypto_data['sma']), 'signal'] += 1
    crypto_data.loc[(crypto_data['macd'] < crypto_data['macd_signal']) & (crypto_data['current_price'] < crypto_data['sma']), 'signal'] -= 1
    crypto_data.loc[(crypto_data['current_price'] > crypto_data['ma200']), 'signal'] += 1
    crypto_data.loc[(crypto_data['current_price'] < crypto_data['ma200']), 'signal'] -= 1
    crypto_data.loc[(crypto_data['pct_change_5h'] > 0.05), 'signal'] += 1
    crypto_data.loc[(crypto_data['pct_change_5h'] < -0.05), 'signal'] -= 1
    crypto_data['days_profit'] = (crypto_data['current_price'] / crypto_data['daily_return']) - 1
    crypto_data.loc[(crypto_data['daily_profit'] > crypto_data['take_profit_value']) | (crypto_data['days_profit'] > crypto_data['take_profit_value']), 'signal'] -= 1
    crypto_data.loc[(crypto_data['daily_profit'] < crypto_data['stop_loss_value']) | (crypto_data['days_profit'] < crypto_data['stop_loss_value']), 'signal'] -= 1
    crypto_data.to_csv('crypto_data.csv', index=False)
    return crypto_data
crypto_data = get_crypto_data(cryptos, crypto_data)
crypto_data.to_csv('crypto_data.csv', index=False)
crypto_data.head()
dir(r.orders)
import time
import datetime
from ratelimit import limits, sleep_and_retry
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
    quantity_or_price = str(quantity_or_price)
    try:
        r.orders.order_crypto(
            symbol=str(symbol),
            quantityOrPrice=float(quantity_or_price),
            amountIn=str(amount_in),
            side=str(side),
            timeInForce='gtc',
            jsonify=True
        )
    except Exception as e:
        raise e
buying_limiter = 0.60
crypto_data = pd.read_csv('crypto_data.csv', converters={'historical_prices': eval})
buying_power = float(r.profiles.load_account_profile(info='crypto_buying_power')) * float(buying_limiter)
def is_daytime():
    current_time = datetime.datetime.now()
    if current_time.hour >= 11 and current_time.hour <= 23:
        return True
    else:
        return False
print(f'I can only play with {buying_power} dollars')
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
        r.cancel_all_crypto_orders()
        time.sleep(10)
        current_time = datetime.datetime.now()
        signal = 0
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
                signal = row['signal'] if row['signal'] != 'nan' else 0
                current_price = float(row['current_price'])
                buying_power = float(row['buying_power'])
                coin_holdings = float(row['coin_holdings'])
                crypto_positions = coin_holdings
                print(f' >> {row["ticker"]} << signal: {row["signal"]}, holding: {row["coin_holdings"]}, ${buying_power}')
                if coin_holdings > 0 and signal < 0:
                    quantity = 0.7 * coin_holdings if signal == -1 else coin_holdings
                    order_crypto(symbol=coin, quantity_or_price=quantity, amount_in='quantity', side='sell')
                    buying_power += current_price * quantity
                    logger.info(Fore.RED + f' [!] ---> Selling {quantity} {coin} at {current_price}...' + Fore.RESET)
                elif signal > 0:
                    amount = 0.005 * buying_power if signal == 1 else (0.01 * (signal - 1) + 0.01) * buying_power
                    order_crypto(symbol=coin, quantity_or_price=amount, amount_in='dollars', side='buy')
                    buying_power -= amount
                    logger.info(Fore.GREEN + f' [!] ---> Buying {amount} {coin} at {current_price}...' + Fore.RESET)
                list_prices = row['historical_prices']
                list_prices = ast.literal_eval(list_prices)
                list_five_latest = list_prices[-5:]
                minimum = min(list_five_latest)
                if isinstance(minimum, str):
                    minimum = float(minimum)
                    logging.info(f'Converting minimum: {minimum} to float... it was a string')
                else:
                    try:
                        minimum = float(minimum)
                        logging.info(f'Converting minimum: {minimum} to float... it was a string')
                    except Exception as e:
                        logging.info(f'Could not convert minimum: {minimum} to float... it was a string')
                        raise e
                if current_price < minimum:
                    if coin_holdings > 0:
                        order_crypto(symbol=coin, quantity_or_price=coin_holdings, amount_in='quantity', side='sell')
                        buying_power += current_price * coin_holdings
                        logger.info(Fore.RED + f' [!] ---> Selling {coin_holdings} {coin} at {current_price}...' + Fore.RESET)
                    else:
                        dollars_spent = 0.005 * buying_power
                        order_crypto(symbol=coin, quantity_or_price=dollars_spent, amount_in='dollars', side='buy')
                        buying_power -= dollars_spent
                        logger.info(Fore.GREEN + f' [!] ---> Buying {dollars_spent} {coin} at {current_price}...' + Fore.RESET)
            tqdm.write(f'we have {buying_power} left')
        except Exception as e:
            logger.error(f"An error occurred: {e}, {traceback.format_exc()}, {dir(e)}")
        print(f'Current buying power: {r.profiles.load_account_profile(info="crypto_buying_power")}')
        print(f'Current profits: {r.profiles.load_account_profile(info="crypto_day_traded_profit_loss")}')
        wait_time = 5 if is_daytime() else 30
        print(f'Waiting {wait_time} minutes...')
        for i in tqdm(range(60*wait_time)):
            time.sleep(1)
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
main_loop()
