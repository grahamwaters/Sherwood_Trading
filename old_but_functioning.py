import logging
import pandas as pd
import pandas_ta as ta
import robin_stocks as rstocks
from robin_stocks import robinhood as r
from datetime import datetime
from pytz import timezone
import asyncio
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import argrelextrema
import time
from pyti.relative_strength_index import relative_strength_index as rsi
from tqdm import tqdm
from icecream import ic
from colorama import Fore, Back, Style
# import ip
import configparser
#^ Starting out with the basics
config = configparser.ConfigParser()
config.read('config/credentials.ini')
coins = config['trading']['coins'].split(', ')
stop_loss_percent = float(config['trading']['stop_loss_percent'])
coins = [coin.strip() for coin in coins]
percent_to_use = float(config['trading']['percent_to_use'])
verbose_mode = config['logging']['verbose_mode']
debug_verbose = config['logging']['debug_verbose']
reset_positions = config['logging']['reset_positions']
minimum_usd_per_position = float(config['trading']['minimum_usd_per_position'])
# minimum_usd_to_use = float(config['trading']['minimum_usd_to_use'])
over_ride = config['trading']['over_ride']

#^ Logging
# Set up logging
logger = logging.getLogger('trader')
logger.setLevel(logging.INFO)
# Create file handler which logs even debug messages
fh = logging.FileHandler('logs/robinhood.log')
fh.setLevel(logging.DEBUG)
# Create console handler with a higher log level
ch = logging.StreamHandler()
if verbose_mode == 'True':
    ch.setLevel(logging.INFO)
else:
    ch.setLevel(logging.WARNING)
# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(ch)
logger.addHandler(fh)

class Utility:
    def __init__(self):
        """
        The Utility class provides utility functions such as getting historical data and checking if it's daytime.
        :doc-author: Trelent
        """
        pass
    async def log_file_size_checker():
        """
        The log_file_size_checker function is an async function that checks the size of the log file and removes lines from the start of the file to maintain a rolling log of 1000 lines.
        :return: None
        :doc-author: Trelent
        """
        while True:
            #ic()
            with open('logs/robinhood.log', 'r') as f:
                lines = f.readlines()
                if len(lines) > 1000: # if the log file is greater than 1000 lines
                    # find how many lines to remove
                    num_lines_to_remove = len(lines) - 1000
                    # remove the first num_lines_to_remove lines
                    with open('logs/robinhood.log', 'w') as f:
                        f.writelines(lines[num_lines_to_remove:])
            await asyncio.sleep(1200)
    def get_last_100_days(self, coin):
        """
        The get_last_100_days function gets the last 100 days of a particular coin's data.
        :param coin: The coin to get data for
        :return: A DataFrame with the last 100 days of coin data
        :doc-author: Trelent
        """
        try:
            df = pd.DataFrame(r.crypto.get_crypto_historicals(coin, interval='hour', span='3month', bounds='24_7'))
            df = df.set_index('begins_at')
            df.index = pd.to_datetime(df.index)
            df = df.loc[:, ['close_price', 'open_price', 'high_price', 'low_price']]
            df = df.rename(columns={'close_price': 'close', 'open_price': 'open', 'high_price': 'high', 'low_price': 'low'})
            df = df.apply(pd.to_numeric)
            return df
        except Exception as e:
            print(f'Unable to get data for {coin}... {e}')
            return pd.DataFrame()
    def is_daytime(self):
        """
        The is_daytime function checks if the current time is between 8 AM and 8 PM.
        :return: True if it's between 8 AM and 8 PM, False otherwise
        :doc-author: Trelent
        """
        current_time = datetime.now(timezone('US/Central'))
        current_hour = current_time.hour
        if current_hour >= 8 and current_hour <= 20:
            return True
        else:
            return False
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
    def resetter(self):
        """
        The resetter function cancels all open orders and sells all positions.
        :doc-author: Trelent
        """
        try:
            open_orders = r.get_all_open_crypto_orders()
            print(Fore.YELLOW + 'Canceling all open orders...' + Style.RESET_ALL)
            for order in tqdm(open_orders):
                r.cancel_crypto_order(order['id'])
            print(Fore.GREEN + 'All open orders cancelled.')
            self.logger.info('All open orders cancelled.' + Style.RESET_ALL)
            crypto_positions = r.get_crypto_positions()
            for position in crypto_positions:
                r.order_sell_crypto_limit(position['currency']['code'], position['quantity'], position['cost_bases'][0]['direct_cost_basis'])
            self.logger.info('All positions sold.')
        except Exception as e:
            self.logger.error(f'Unable to reset orders and positions... {e}')

    def rsi(self, df, n):
        """
        The rsi function calculates the Relative Strength Index (RSI) for a given DataFrame and time period.
        :param df: The DataFrame to calculate the RSI for
        :param n: The time period to calculate the RSI for
        :return: The RSI for the given DataFrame and time period
        :doc-author: Trelent
        """
        try:
            df = float(df['close'])
            delta = df.diff()
            delta = delta[1:]
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0
            roll_up = up.ewm(com=n - 1, min_periods=n).mean()
            roll_down = down.abs().ewm(com=n - 1, min_periods=n).mean()
            rs = roll_up / roll_down
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi = rsi.iloc[-1]
            rsi = float(rsi)
            return rsi

        except Exception as e:
            self.logger.error(f'Unable to calculate RSI... {e}')
    def log_file_size_checker(self):
        """
        The log_file_size_checker function checks the size of the log file. If the size is greater than 1 MB, the log file is cleared.
        :doc-author: Trelent
        """
        try:
            if os.path.getsize('logs/trader.log') > 1000000:
                open('logs/trader.log', 'w').close()
        except Exception as e:
            self.logger.error(f'Unable to check the size of the log file... {e}')

    def calculate_ta_indicators(self, coins):
        """
        The calculate_ta_indicators function calculates different technical indicators and generates trading signals based on these indicators. The indicators are: EMA, MACD, RSI, Williams %R, Stochastic Oscillator, Bollinger Bands, and Parabolic SAR.
        A boolean is generated based on these indicators. If the boolean is True, a buy signal is generated. If the boolean is False, a sell signal is generated. The signals are returned in a DataFrame.
        :param coins: A list of coins to generate signals for
        :return: A DataFrame with the trading signals for each coin
        """
        try:
            utility = Utility()
            signals_df = pd.DataFrame()
            for coin in coins:
                df = utility.get_last_100_days(coin)
                df['sma'] = df.close.rolling(window=50).mean()
                df['ema'] = df.close.ewm(span=50, adjust=False).mean()
                macd_line = ta.macd(df.close)['MACD_12_26_9'][-1]
                #^ The MACD object is a tuple with the MACD, signal line, and histogram values
                macd_signal = float(ta.macd(df.close)['MACDs_12_26_9'][-1])

                df['rsi'] = self.rsi(df, 14)
                ic()
                df['williams'] = float(ta.willr(df.high, df.low, df.close)[-1])
                ic()
                # df['stochastic_k'], df['stochastic_d'] = ta.stoch(df.high, df.low, df.close)
                #
                # bollinger_upper_band = config['trading']['bollinger_upper_band']
                # bollinger_lower_band = config['trading']['bollinger_lower_band']
                # df['bollinger_middle_band'] = df['close'].rolling(window=20).mean()
                df['bollinger_upper_band'] = ta.bbands(df.close)['BBU_5_2.0'][-1]
                df['bollinger_lower_band'] = ta.bbands(df.close)['BBL_5_2.0'][-1]
                signals_df = signals_df.append(df)

                #^ Generate the buy signals
                # cast the values of close, sma, and ema to floats to avoid the error: "The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()."


                # Assuming signals_df and df are DataFrame objects, and macd_line and macd_signal are also defined somewhere in your code.

                # Cast to floats and add conditions
                signals_df['buy_signal'] = (
                    (float(df['ema'])) > float(df['sma']) &
                    (float(df['rsi']) < 30) &
                    (float(df['williams']) < -80) &
                    (float(df['close']) < float(df['bollinger_lower_band'])) &
                    (float(df['close']) < float(df['sma'])) &
                    (float(df['close']) < float(df['ema'])) &
                    (float(float(macd_line)) > float(macd_signal))
                )
                # Cast to floats and add conditions
                signals_df['sell_signal'] = (
                    (float(df['ema'])) < float(df['sma']) &
                    (float(df['rsi']) > 70) &
                    (float(df['williams']) > -20) &
                    (float(df['close']) > float(df['bollinger_upper_band'])) &
                    (float(df['close']) > float(df['sma'])) &
                    (float(df['close']) > float(df['ema'])) &
                    (float(float(macd_line)) < float(macd_signal))
                )
                # Drop the NaNs from the DataFrame
                signals_df = signals_df.dropna()

                #^ Generate the MACD line and signal line
                signals_df['macd_line'] = macd_line
                signals_df['signal_line'] = macd_signal

            return signals_df
        except Exception as e:
            self.logger.error(f'Unable to generate trading signals... {e}')
            return pd.DataFrame()
    def trading_function(self, signals_df):
        """
        The trading_function function takes the trading signals generated by calculate_ta_indicators() and places trades accordingly.
        :param signals_df: A DataFrame with the trading signals for each coin
        :doc-author: Trelent
        """
        try:
            crypto_positions = r.get_crypto_positions()
            for index, row in signals_df.iterrows():
                if row['buy_signal']:
                    #* Create a nice little data viz block for the terminal that shows the TA indicators for why this position was bought
                    block_text = f"""
                    {row['coin']} bought at {row['close']} because:
                    - MACD Line: {row['macd_line']}
                    - Signal Line: {row['signal_line']}
                    - RSI: {row['rsi']}
                    - Williams %R: {row['williams']}
                    - Stochastic K: {row['stochastic_k']}
                    - Stochastic D: {row['stochastic_d']}
                    """
                    print(Fore.GREEN + block_text + Style.RESET_ALL)
                    # Check if we have enough buying power to buy this coin
                    buying_power = self.update_buying_power()
                    if buying_power > 0:
                        r.order_buy_crypto_limit(symbol=row['coin'],
                                                    quantity = buying_power / row['close'],
                                                    limitPrice = row['close'],
                                                    timeInForce = 'gtc')
                        self.logger.info(f'Bought {row["coin"]} at {row["close"]}.')
                if row['sell_signal']:
                    for position in crypto_positions:
                        #* Create a nice little data viz block for the terminal that shows the TA indicators for why this position was sold
                        block_text = f"""
                        {row['coin']} sold at {row['close']} because:
                        - MACD Line: {row['macd_line']}
                        - Signal Line: {row['signal_line']}
                        - RSI: {row['rsi']}
                        - Williams %R: {row['williams']}
                        - Stochastic K: {row['stochastic_k']}
                        - Stochastic D: {row['stochastic_d']}
                        """
                        print(Fore.RED + block_text + Style.RESET_ALL)
                        if position['currency']['code'] == row['coin']:
                            r.order_sell_crypto_limit(symbol=row['coin'],
                                                        quantity=position['quantity'],
                                                        limitPrice=row['close'],
                                                        timeInForce='gtc')
                            self.logger.info(f'Sold {row["coin"]} at {row["close"]}.')
        except Exception as e:
            self.logger.error(f'Unable to execute trades... {e}')
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
            profile_info = r.load_account_profile()
            cash_available = float(profile_info['cash_available_for_withdrawal'])
            crypto_dollars = self.get_total_crypto_dollars()
            buying_power = cash_available + crypto_dollars
            return buying_power
        except Exception as e:
            self.logger.error(f'Unable to update buying power... {e}')
            return 0
    # def check_stop_loss_prices(self, coins, stop_loss_prices):
    #     """
    #     The check_stop_loss_prices function checks if the current price is lower than the stop loss price for any owned coin.
    #     :param coins: A list of coins to check
    #     :param stop_loss_prices: A dictionary with the stop loss price for each coin
    #     :doc-author: Trelent
    #     """
    #     try:
    #         for coin in tqdm(coins):
    #             crypto_positions = r.get_crypto_positions()
    #             for position in crypto_positions:
    #                 if position['currency']['code'] == coin:
    #                     current_price = float(r.crypto.get_crypto_quote(coin)['mark_price'])
    #                     if current_price < stop_loss_prices[coin]:
    #                         r.orders.order_sell_crypto_limit(
    #                             symbol=coin,
    #                             quantity=position['quantity'],
    #                             limitPrice=current_price,
    #                             timeInForce='gtc')
    #                         self.logger.info(f'Sold {coin} at {current_price} due to stop loss.')
    def main(self, coins, stop_loss_prices):
        """
        The main function is the main function. It will do the following:
            1) Check if there are any open orders that need to be cancelled
            2) Check if there are any positions that need to be sold (if we're in a sell state)
            3) If we're in a buy state, check for new stocks to buy based on our criteria
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :return: The main function
        :doc-author: Trelent
        """
        try:
            utility = Utility()
            if utility.is_daytime() or over_ride:
                self.resetter()
                signals_df = self.calculate_ta_indicators(coins)
                self.trading_function(signals_df)
                # self.check_stop_loss_prices(coins, stop_loss_prices)
            else:
                self.logger.info('It is not daytime. The main function will not run.')
        except Exception as e:
            self.logger.error(f'Unable to run main function... {e}')
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
        """
        The run_async_functions function is the main function that runs all of the other functions.
        It will run them simultaneously, and it will also keep track of how many times it has looped.
        The loop_count variable is used to determine when to update buying power, which happens every 10 loops.
        :param loop_count: Keep track of how many times the loop has run
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :return: A coroutine object
        :doc-author: Trelent
        """
        try:
            if loop_count % 10 == 0:
                self.trader.update_buying_power()
            self.trader.main(coins, stop_loss_prices)
            # run all async functions simultaneously
            # log_file_size_checker included to prevent log file from getting too large
            self.trader.log_file_size_checker()
        except Exception as e:
            self.logger.error(f'Unable to run async functions... {e}')
    async def main_looper(self, coins, stop_loss_prices):
        """
        The main_looper function is the main loop function. It will run every hour and do the following:
            1) Check if there are any open orders that need to be cancelled
            2) Check if there are any positions that need to be sold (if we're in a sell state)
            3) If we're in a buy state, check for new stocks to buy based on our criteria
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :doc-author: Trelent
        """
        loop_count = 0
        while True:
            try:
                await self.run_async_functions(loop_count, coins, stop_loss_prices)
                loop_count += 1
                await asyncio.sleep(3600)  # Sleep for an hour
            except Exception as e:
                self.logger.error(f'Error in main loop... {e}')
# run the program
if __name__ == '__main__':
    stop_loss_percent = 0.05 #^ set the stop loss percent at 5% (of the invested amount)
    coins = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']

    #^ set stop losses for each coin by multiplying the current price by the stop loss percent (0.05) and subtracting that from the current price (to get the stop loss price).
    config = configparser.ConfigParser()
    config.read('config/credentials.ini')
    username = config['robinhood']['username']
    password = config['robinhood']['password']
    login = r.login(username, password)
    print(f"Logged in as {username}")
    stop_loss_prices = {
        "BTC": float(r.crypto.get_crypto_quote("BTC")['mark_price']) * (1 - stop_loss_percent),
        "ETH": float(r.crypto.get_crypto_quote("ETH")['mark_price']) * (1 - stop_loss_percent),
        "DOGE": float(r.crypto.get_crypto_quote("DOGE")['mark_price']) * (1 - stop_loss_percent),
        "SHIB": float(r.crypto.get_crypto_quote("SHIB")['mark_price']) * (1 - stop_loss_percent),
        "ETC": float(r.crypto.get_crypto_quote("ETC")['mark_price']) * (1 - stop_loss_percent),
        "UNI": float(r.crypto.get_crypto_quote("UNI")['mark_price']) * (1 - stop_loss_percent),
        "AAVE": float(r.crypto.get_crypto_quote("AAVE")['mark_price']) * (1 - stop_loss_percent)
    }
    trader = Trader(
        username = username,
        password = password
        ) #^ create an instance of the Trader class
    looper = Looper(trader) #^ create an instance of the Looper class (which will run the Trader class)
    asyncio.run(looper.main_looper(coins, stop_loss_prices)) #^ run the main_looper function