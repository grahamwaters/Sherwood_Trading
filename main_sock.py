import logging
import pandas as pd
import pandas_ta as ta
import robin_stocks as rstocks
from robin_stocks import robinhood as r
from datetime import datetime
from pytz import timezone
import asyncio
from tqdm import tqdm
from colorama import Fore, Style
import os
import time
import json
import numpy as np

#^Formatting Options
# make all float point numbers display to 9 decimal places
pd.set_option('display.float_format', lambda x: '%.9f' % x)
# make all columns display
pd.set_option('display.max_columns', None)

#^Global Variables
stop_loss_percent = 0.05 # The percentage of loss we are willing to take before selling
verbose_mode = False # Set to True to see all the logging messages
percent_to_use = 0.60 # The percentage of our portfolio we are willing to gamble with (i.e. buy crypto with during the session)

class Utility:
    def __init__(self):
        self.logger = logging.getLogger('Utility')

    async def log_file_size_checker(self):
        """
        The log_file_size_checker function is a coroutine that checks the size of the log file.
        If it exceeds 1000 lines, it will remove all but the last 1000 lines.

        :param self: Represent the instance of the object itself
        :return: A coroutine object, which is an awaitable
        :doc-author: Trelent
        """

        while True:
            log_file = 'logs/robinhood.log'
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1000:
                        num_lines_to_remove = len(lines) - 1000
                        with open(log_file, 'w') as f:
                                                        f.writelines(lines[num_lines_to_remove:])
            await asyncio.sleep(1200)

    def get_last_100_days(self, coin):
        try:
            df = pd.DataFrame(r.crypto.get_crypto_historicals(coin, interval='hour', span='3month', bounds='24_7'))
            df = df.set_index('begins_at')
            df.index = pd.to_datetime(df.index)
            df = df.loc[:, ['close_price', 'open_price', 'high_price', 'low_price']]
            df = df.rename(columns={'close_price': 'close', 'open_price': 'open', 'high_price': 'high', 'low_price': 'low'})
            df = df.apply(pd.to_numeric)
            return df
        except Exception as e:
            self.logger.error(f'Unable to get data for {coin}... {e}')
            return pd.DataFrame()

    def is_daytime(self):
        """
        The is_daytime function checks the current time and returns a boolean value.
            If it is between 8am and 8pm, then it will return True. Otherwise, False.

        :param self: Allow an object to refer to itself inside of a method
        :return: True if the current time is between 8am and 8pm
        :doc-author: Trelent
        """

        current_time = datetime.now(timezone('US/Central'))
        current_hour = current_time.hour
        return 8 <= current_hour <= 20


class Batch:
    def __init__(self, coin, purchase_price, quantity):
        self.coin = coin
        self.purchase_price = purchase_price
        self.quantity = quantity
        self.stop_loss_price = purchase_price * (1 - stop_loss_percent)

    def check_stop_loss(self, current_price):
        """
        The check_stop_loss function is used to determine if the current price of a stock has fallen below the stop loss price.
            If it has, then we will sell our shares and take a loss.

        :param self: Represent the instance of the class
        :param current_price: Check if the current price is less than the stop loss price
        :return: A boolean value
        :doc-author: Trelent
        """

        return current_price < self.stop_loss_price


class Trader:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.logger = logging.getLogger('trader')
        self.setup_logging()
        self.login_setup()
        self.batches = []

    def setup_logging(self):
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def login_setup(self):
        try:
            r.login(self.username, self.password)
            self.logger.info('Logged in to Robinhood successfully.')
        except Exception as e:
            self.logger.error(f'Unable to login to Robinhood... {e}')

    def get_total_crypto_dollars(self):
        """
        The get_total_crypto_dollars function calculates the total value of all crypto owned.
        :return: The total value of all crypto owned
        :doc-author: Trelent
        """
        try:
            crypto_positions = r.get_crypto_positions()
            total_crypto_dollars = 0
            for position in crypto_positions:
                total_crypto_dollars += float(position['quantity']) * float(r.crypto.get_crypto_quote(position['currency']['code'])['mark_price'])
            return total_crypto_dollars
        except Exception as e:
            self.logger.error(f'Unable to get total value of crypto... {e}')
            return 0

    def update_buying_power(self):
        """
        The update_buying_power function updates the buying power of the user's account.
        :return: The updated buying power
        :doc-author: Trelent
        """
        try:
            profile_info = r.load_account_profile() #^ This is the profile information for the user
            cash_available = float(profile_info['cash_available_for_withdrawal']) #^ This is the amount of cash available to spend
            crypto_dollars = self.get_total_crypto_dollars() #^ This is the total value of all crypto in dollars
            buying_power = cash_available + crypto_dollars # This is the total buying power
            return buying_power * percent_to_use
        except Exception as e:
            self.logger.error(f'Unable to update buying power... {e}')
            return 0
    def resetter(self):
        try:
            open_orders = r.get_all_open_crypto_orders()
            self.logger.info('Canceling all open orders...')
            for order in open_orders:
                r.cancel_crypto_order(order['id'])
            self.logger.info('All open orders cancelled.')
            crypto_positions = r.get_crypto_positions()
            for position in crypto_positions:
                r.order_sell_crypto_limit(position['currency']['code'], position['quantity'], position['cost_bases'][0]['direct_cost_basis'])
            self.logger.info('All positions sold.')
        except Exception as e:
            self.logger.error(f'Unable to reset orders and positions... {e}')

    def calculate_ta_indicators(self, coins):
        try:
            utility = Utility()
            signals = []
            for coin in coins:
                df = utility.get_last_100_days(coin)
                df['sma'] = df.close.rolling(window=50).mean()
                df['ema'] = df.close.ewm(span=50, adjust=False).mean()
                df['macd_line'], df['signal_line'], df['macd_hist'] = ta.macd(df.close)
                df['rsi'] = ta.rsi(df.close)
                df['buy_signal'] = ((df.macd_line > df.signal_line) & (df.rsi < 30)) #^ this indicates that the macd line has crossed above the signal line, and the rsi is below 30 --> buy
                df['sell_signal'] = ((df.macd_line < df.signal_line) & (df.rsi > 70)) #^ this indicates that the macd line has crossed below the signal line, and the rsi is above 70 --> sell
                signals.append({
                    'coin': coin,
                    'buy_signal': df['buy_signal'].iat[-1],
                    'sell_signal': df['sell_signal'].iat[-1],
                    'close': df['close'].iat[-1],
                    'indicators': df.iloc[-1].to_dict()
                })
            return pd.DataFrame(signals)
        except Exception as e:
            self.logger.error(f'Unable to generate trading signals... {e}')
            print(f'LOGGER ERROR: {e}')
            return pd.DataFrame()

    def trading_function(self, signals_df):
        #^ first print how much we have in buying power, and how much we have in crypto (by value)
        buying_power = self.update_buying_power() #^ this is the amount of buying power we have
        crypto_value = self.get_total_crypto_dollars() #^ this is the value of all crypto in dollars
        print(Fore.GREEN + f'Buying power: ${buying_power:.2f}' + Style.RESET_ALL)
        print(Fore.GREEN + f'Crypto value: ${crypto_value:.2f}' + Style.RESET_ALL)
        print(Fore.GREEN + f'Total value: ${buying_power + crypto_value:.2f}' + Style.RESET_ALL)
        print(f'-'*20)

        try:
            crypto_positions = {position['currency']['code']: position for position in r.get_crypto_positions()}
            for _, row in signals_df.iterrows():
                coin = row['coin']
                close_price = row['close']
                indicators = row['indicators']
                if row['buy_signal']:
                    if verbose_mode:
                        self.logger.info(Fore.GREEN + f'Bought {coin} at {close_price} because:\n' + '\n'.join(f'- {key}: {value}' for key, value in indicators.items()) + Style.RESET_ALL)
                    else:
                        self.logger.info(f'Bought {coin} at ${close_price}')
                    buying_power = self.update_buying_power()
                    # We want to only use percent_to_use % of our buying power for the sessions trades. This keeps 100-percent_to_use % back as gaurenteed profit.
                    buying_power = buying_power * percent_to_use

                    if buying_power > 0:
                        quantity = buying_power / close_price
                        r.order_buy_crypto_limit(symbol=coin, quantity=quantity, limitPrice=close_price, timeInForce='gtc')
                        self.batches.append(Batch(coin, close_price, quantity)) #note: for gpt agents, the line here is contingent upon the line above being successful. We can't know that it is successful until we check the order status, later  on. So, for future refactorizations we should consider adapting this method of batch creation to include some sort of check to ensure that the order was successful before appending the batch to the list.
                if row['sell_signal'] and coin in crypto_positions:
                    if verbose_mode:
                        self.logger.info(f'Sold {coin} at {close_price} because:\n' + '\n'.join(f'- {key}: {value}' for key, value in indicators.items()))
                    else:
                        self.logger.info(Fore.RED + f'Sold {coin} at ${close_price}' + Style.RESET_ALL)
                    position = crypto_positions[coin]
                    r.order_sell_crypto_limit(symbol=coin, quantity=position['quantity'], limitPrice=close_price, timeInForce='gtc')
                    self.batches = [batch for batch in self.batches if batch.coin != coin]
        except Exception as e:
            self.logger.error(f'Unable to execute trades... {e}')

    def check_stop_loss_prices(self):
        try:
            for batch in self.batches:
                current_price = float(r.crypto.get_crypto_quote(batch.coin)['mark_price'])
                if batch.check_stop_loss(current_price):
                    r.order_sell_crypto_limit(batch.coin, batch.quantity, current_price)
                    self.logger.info(f'Sold {batch.coin} at {current_price} due to stop loss.')
                    self.batches.remove(batch)
        except Exception as e:
            self.logger.error(f'Unable to check stop loss prices... {e}')

with open('config/credentials.json') as f:
    config = json.load(f)
    username = config['username']
    password = config['password']
    coins = config['coins']


if __name__ == '__main__':
    trader = Trader(username, password)
    trader.resetter()
    while True:
        signals_df = trader.calculate_ta_indicators(coins)
        trader.trading_function(signals_df)
        trader.check_stop_loss_prices()
        time.sleep(60)
