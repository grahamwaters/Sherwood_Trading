import logging
import pandas as pd
import pandas_ta as ta
import requests
import robin_stocks as rstocks
from robin_stocks import robinhood as r
from datetime import datetime
from pytz import timezone
import asyncio
import time
import configparser
import json
import os
from tqdm import tqdm
from colorama import Fore, Back, Style
from ratelimit import limits, sleep_and_retry

#^ Global Variables
LOG_FILE = 'logs/crypto.log'
LOG_FILE_SIZE_LIMIT = 50000000  # 50 MB


class Utility:
    def __init__(self):
        """
        The Utility class provides utility functions such as getting historical data and checking if it's daytime.
        :doc-author: Trelent
        """
        # Set up logging
        self.logger = logging.getLogger('utility')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        pass

    #*fixed
    def get_last_100_days(self, coin):
        try:
            df = self.get_historical_data(coin)
            df = self.process_historical_data(df)
            return df
        except Exception as e:
            self.logger.error(f'Unable to get historical data for {coin}... {e}')
            return pd.DataFrame()
    #*fixed
    def get_historical_data(self, coin):
        #^ Fixed ->'robin_stocks.robinhood' has no attribute 'get_historicals'
        return pd.DataFrame(r.get_crypto_historicals(coin,
                                                     interval='hour', span='3month'))
    #*fixed
    def process_historical_data(self, df):
        df = df.set_index('begins_at')
        df.index = pd.to_datetime(df.index)
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
        df = df[df.volume != '0']
        df = df.iloc[-100:]
        return df

    def is_daytime(self):
        time_now = datetime.now().time()
        return time(6, 0) <= time_now <= time(20, 0)

    def get_coins(self):
        try:
            return self.get_tradable_coins()
        except Exception as e:
            self.logger.error(f'Unable to get list of coins... {e}')
            return []

    def get_tradable_coins(self):
        coins = r.get_crypto_info()
        return [coin['symbol'] for coin in coins if coin['tradability'] == 'tradable']


class Trader:
    """
    The Trader class provides functions for logging into Robinhood, resetting orders, generating trading signals, executing actions based on these signals, updating the buying power, and checking stop loss prices.
    # Detailed Function Descriptions
    1. login_setup: The login_setup function logs into Robinhood using the provided username and password.
    2. resetter: The resetter function cancels all open orders and sells all positions. This function is used to reset the bot.
    3. calculate_ta_indicators:
        The calculate_ta_indicators function calculates different technical indicators and generates trading signals based on these indicators. The indicators are: EMA, MACD, RSI, Williams %R, Stochastic Oscillator, Bollinger Bands, and Parabolic SAR.
        A boolean is generated based on these indicators. If the boolean is True, a buy signal is generated. If the boolean is False, a sell signal is generated. The signals are returned in a DataFrame.
        :param coins: A list of coins to generate signals for
        :return: A DataFrame with the trading signals for each coin
    4. action_module:
        The action_module function executes actions based on the trading signals. If the signal is a buy signal, the coin is bought. If the signal is a sell signal, the coin is sold, if it is owned.
        :param signals: A DataFrame with the trading signals for each coin
        :return: None
    5. buying_power_updater:
        The buying_power_updater function updates the buying power of the Robinhood account (in USD).
        :return: None
    6. stop_loss_checker:
        The stop_loss_checker function checks if the current price of a coin is below the stop loss price. If it is, the coin is sold.
        :param coins: A list of coins to check the stop loss price for
        :return: None
    """
    def __init__(self, username, password):
        """
        The Trader class provides functions for logging into Robinhood, resetting orders, generating trading signals,
        executing actions based on these signals, updating the buying power, and checking stop loss prices.
        :param username: The username for the Robinhood account
        :param password: The password for the Robinhood account
        :doc-author: Trelent
        """
        self.username = username
        self.password = password
        # Set up logging
        self.logger = logging.getLogger('trader')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # Login to Robinhood
        self.login_setup()
    def login_setup(self):
        """
        The login_setup function logs into Robinhood using the provided username and password.
        :doc-author: Trelent
        """
        try:
            r.login(self.username, self.password)
            self.logger.info('Logged in to Robinhood successfully.')
        except Exception as e:
            self.logger.error(f'Unable to login to Robinhood... {e}')
    def cancel_open_orders(self):
        try:
            open_orders = r.get_all_open_crypto_orders()
            print(Fore.YELLOW + 'Canceling all open orders...' + Style.RESET_ALL)
            for order in tqdm(open_orders):
                r.cancel_crypto_order(order['id'])
            print(Fore.GREEN + 'All open orders cancelled.')
            self.logger.info('All open orders cancelled.' + Style.RESET_ALL)
        except Exception as e:
            self.logger.error(f'Unable to cancel open orders... {e}')

    def sell_all_positions(self):
        try:
            crypto_positions = r.get_crypto_positions()
            for position in crypto_positions:
                r.order_sell_crypto_limit(position['currency']['code'], position['quantity'], position['cost_bases'][0]['direct_cost_basis'])
            self.logger.info('All positions sold.')
        except Exception as e:
            self.logger.error(f'Unable to sell all positions... {e}')

    #*fixed
    def resetter(self):
        """
        The resetter function cancels all open orders and sells all positions. This function is used to reset the bot.
        :doc-author: Trelent
        """
        self.cancel_open_orders()
        self.sell_all_positions()


    async def calculate_ta_indicators(self, coins):
        signals_df = pd.DataFrame()
        utility = Utility()
        for coin in coins:
            try:
                df = await self.calculate_single_coin_indicators(coin, utility)
                signals_df = signals_df.append(df)
            except Exception as e:
                self.logger.error(f'Unable to generate trading signals for {coin}... {e}')
        return signals_df

    async def calculate_single_coin_indicators(self, coin, utility):
        df = utility.get_last_100_days(coin)
        df['sma'] = df.close.rolling(window=50).mean()
        df['ema'] = df.close.ewm(span=50, adjust=False).mean()
        df['macd_line'], df['signal_line'], df['macd_hist'] = ta.macd(df.close)
        df['rsi'] = ta.rsi(df.close)
        df['williams'] = ta.williams_r(df.high, df.low, df.close)
        df['stochastic_k'], df['stochastic_d'] = ta.stoch(df.high, df.low, df.close)
        df['bollinger_l'], df['bollinger_m'], df['bollinger_u'] = ta.bollinger_bands(df.close)
        df['buy_signal'] = ((df.macd_line > df.signal_line) & (df.rsi < 30)) | ((df.stochastic_k > df.stochastic_d) & (df.williams < -80))
        df['sell_signal'] = ((df.macd_line < df.signal_line) & (df.rsi > 70)) | ((df.stochastic_k < df.stochastic_d) & (df.williams > -20))
        return df

    async def trading_function(self, coins, signals_df):
        for coin in coins:
            try:
                self.handle_buy_signals(coin, signals_df)
                self.handle_sell_signals(coin, signals_df)
            except Exception as e:
                self.logger.error(f'Unable to execute trading function for {coin}... {e}')

    def handle_buy_signals(self, coin, signals_df):
        if 'buy_signal' in signals_df.columns and 'close' in signals_df.columns and signals_df.at[coin, 'buy_signal'] and self.buying_power > signals_df.at[coin, 'close']:
            r.order_buy_crypto_limit(coin, self.buying_power / signals_df.at[coin, 'close'], signals_df.at[coin, 'close'])
            self.buying_power -= signals_df.at[coin, 'close']
            self.logger.info(f'Purchased {coin}.')

    def handle_sell_signals(self, coin, signals_df):
        if 'sell_signal' in signals_df.columns and 'quantity' in signals_df.columns and 'close' in signals_df.columns and signals_df.at[coin, 'sell_signal']:
            r.order_sell_crypto_limit(coin, signals_df.at[coin, 'quantity'], signals_df.at[coin, 'close'])
            self.buying_power += signals_df.at[coin, 'close']
            self.logger.info(f'Sold {coin}.')


    #*fixed
    def get_total_crypto_dollars(self):
        total_crypto_dollars = 0
        crypto_positions = self.get_crypto_positions_safely()
        for position in crypto_positions:
            total_crypto_dollars += self.calculate_position_value(position)
        return total_crypto_dollars

    def get_crypto_positions_safely(self):
        try:
            return r.get_crypto_positions()
        except Exception as e:
            self.logger.error(f'Unable to get crypto positions... {e}')
            return []

    def calculate_position_value(self, position):
        try:
            return float(position['quantity']) * float(position['cost_bases'][0]['direct_cost_basis'])
        except Exception as e:
            self.logger.error(f'Unable to calculate position value... {e}')
            return 0


    #*fixed
    async def update_buying_power(self):
        try:
            #note: this is a valid await
            self.buying_power = await self.get_crypto_buying_power_safely()
            self.logger.info(f'Updated buying power to {self.buying_power}.')
        except Exception as e:
            self.logger.error(f'Unable to update buying power... {e}')

    @sleep_and_retry
    @limits(calls=6, period=60) # 6 calls per 60 seconds
    async def get_crypto_buying_power_safely(self):
        try:
            return float(r.profiles.load_account_profile(info='buying_power'))
        except Exception as e:
            self.logger.error(f'Unable to get crypto buying power... {e}')
            return 0.0

    async def check_stop_loss_prices(self, coins, stop_loss_prices):
        for coin in coins:
            try:
                #note: this is a valid await
                await self.check_stop_loss_price_for_coin(coin, stop_loss_prices)
            except Exception as e:
                self.logger.error(f'Unable to check stop loss price for {coin}... {e}')

    async def check_stop_loss_price_for_coin(self, coin, stop_loss_prices):
        #note: this is a valid await
        latest_price = await self.get_latest_crypto_price_safely(coin)
        if latest_price < stop_loss_prices[coin]:
            #note: this is a valid await
            await self.sell_coin_due_to_stop_loss(coin, latest_price)

    @sleep_and_retry
    @limits(calls=6, period=60) # 6 calls per 60 seconds
    async def get_latest_crypto_price_safely(self, coin):
        try:
            return float(r.crypto.get_crypto_quote(coin)['mark_price'])
        except Exception as e:
            self.logger.error(f'Unable to get latest crypto price for {coin}... {e}')
            return float('inf')

    @sleep_and_retry
    async def sell_coin_due_to_stop_loss(self, coin, latest_price):
        try:
            quantity = r.get_crypto_positions()[coin]['quantity']
            r.order_sell_crypto_limit(coin, quantity, latest_price)
            self.logger.info(f'Sold {coin} due to stop loss price.')
        except Exception as e:
            self.logger.error(f'Unable to sell {coin} due to stop loss... {e}')

    async def main(self, coins, stop_loss_prices):
        try:
            #note: this is a valid await
            await self.start_trading_loop(coins, stop_loss_prices)
        except Exception as e:
            self.logger.error(f'Unable to start trading... {e}')

    async def start_trading_loop(self, coins, stop_loss_prices):
        while True:
            self.resetter()

            await self.update_buying_power()
            signals_df = await self.calculate_ta_indicators(coins)
            await self.trading_function(coins, signals_df)
            await self.check_stop_loss_prices(coins, stop_loss_prices)
            await asyncio.sleep(60)

class Looper:
    def __init__(self, trader: Trader):
        """
        The Looper class provides functions for running asynchronous operations.
        :param trader: An instance of the Trader class
        :doc-author: Trelent
        """
        self.trader = trader
        # Set up logging
        self.logger = logging.getLogger('looper')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    async def run_async_functions(self, loop_count, coins, stop_loss_prices):
        try:
            if loop_count % 10 == 0:
                await self.trader.update_buying_power()
            await self.trader.main(coins, stop_loss_prices)
            self.trader.log_file_size_checker()
        except Exception as e:
            self.logger.error(f'Unable to run async functions... {e}')

    async def main_looper(self, coins, stop_loss_prices):
        loop_count = 0
        while True:
            try:
                await self.run_async_functions(loop_count, coins, stop_loss_prices)
                loop_count += 1
                await asyncio.sleep(3600)  # Sleep for an hour
            except requests.exceptions.RequestException as e:  # Network-related exceptions
                self.logger.error(f'Network error in main loop... {e}')
                await asyncio.sleep(10)  # Short sleep before retry
                continue  # Retry the loop
            except Exception as e:  # Other exceptions
                self.logger.error(f'Error in main loop... {e}')
                break  # Stop the loop

    def log_file_size_checker(self):
        try:
            if self.log_file_exceeds_size_limit():
                os.remove(LOG_FILE)
        except OSError as e:
            self.logger.error(f'Unable to check or delete log file... {e}')

    def log_file_exceeds_size_limit(self):
        file_info = os.stat(LOG_FILE)
        return file_info.st_size >= LOG_FILE_SIZE_LIMIT



def calculate_stop_loss_prices(coins, stop_loss_percent):
    stop_loss_prices = {}
    for coin in coins:
        try:
            mark_price = float(r.crypto.get_crypto_quote(coin)['mark_price'])
            stop_loss_price = mark_price - (mark_price * stop_loss_percent)
            stop_loss_prices[coin] = stop_loss_price
        except Exception as e:
            print(f'Unable to calculate stop loss price for {coin}... {e}')
    return stop_loss_prices

if __name__ == '__main__':
    stop_loss_percent = 0.05  # Set the stop loss percent at 5% (of the invested amount)
    # load the username and password from credentials.ini in the config folder
    config = configparser.ConfigParser()
    config.read('config/credentials.ini')
    username = config['credentials']['username']
    password = config['credentials']['password']
    coins = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']
    trader = Trader(username, password)  # Create an instance of the Trader class
    stop_loss_prices = calculate_stop_loss_prices(coins, stop_loss_percent)
    print(f'Stop loss prices: {stop_loss_prices}')
    looper = Looper(trader)  # Create an instance of the Looper class (which will run the Trader class)
    asyncio.run(looper.main_looper(coins, stop_loss_prices))  # Run the main_looper function
