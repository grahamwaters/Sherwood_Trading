import logging
import configparser
import pandas as pd
import pandas_ta as ta
import robin_stocks as rstocks
from robin_stocks import robinhood as r
from datetime import datetime
from pytz import timezone
import alive_progress
import traceback
from alive_progress import alive_bar
from icecream import ic
import asyncio
import sys
# from legacy.V5.main2 import stop_loss_percent
from tqdm import tqdm
from colorama import Fore, Back, Style
class Utility:
    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger('utility')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
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
        formatter = logging.Formatter(f'{Fore.YELLOW}%(asctime)s{Style.RESET_ALL} - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Add the exception hook to log uncaught exceptions
        sys.excepthook = self.handle_uncaught_exception

        # Login to Robinhood
        self.login_setup()

    def handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """
        Custom exception hook to log uncaught exceptions with traceback.
        """
        self.logger.error("Uncaught exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))


    # async def log_file_size_checker(self):
    #     """
    #     The log_file_size_checker function checks the size of the log file and removes lines from the start of the file to maintain a rolling log of 1000 lines.
    #     :return: None
    #     :doc-author: Trelent
    #     """
    #     while True:
    #         #ic()
    #         with open('logs/robinhood.log', 'r') as f:
    #             lines = f.readlines()
    #             if len(lines) > 1000:
    #                 num_lines_to_remove = len(lines) - 1000
    #                 with open('logs/robinhood.log', 'w') as f:
    #                     f.writelines(lines[num_lines_to_remove:])
    #         await asyncio.sleep(1200)

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
                print(Fore.YELLOW + f'Selling {position["currency"]["code"]} position...' + Style.RESET_ALL)
            self.logger.info('All positions sold.')
        except Exception as e:
            self.logger.error(f'Unable to reset orders and positions... {e}')
    def calculate_ta_indicators(self, coins):
        """
        The calculate_ta_indicators function calculates different technical indicators and generates trading signals based on these indicators. The indicators are: EMA, MACD, RSI, Williams %R, Stochastic Oscillator, Bollinger Bands, and Parabolic SAR.
        A boolean is generated based on these indicators. If the boolean is True, a buy signal is generated. If the boolean is False, a sell signal is generated. The signals are returned in a DataFrame.
        :param coins: A list of coins to generate signals for
        :return: A DataFrame with the trading signals for each coin
        """
        with alive_bar(len(coins)) as living_bar:
            try:
                utility = Utility()
                signals_df = pd.DataFrame()

                for coin in coins:
                    # update the bar on each iteration with the coin name
                    living_bar.text(f'Calculating TA indicators for {coin}...')
                    df = utility.get_last_100_days(coin)
                    df['sma'] = df.close.rolling(window=50).mean()
                    df['ema'] = df.close.ewm(span=50, adjust=False).mean()
                    macd_result = ta.macd(df.close)
                    df['macd_line'] = macd_result['MACD_12_26_9']
                    df['signal_line'] = macd_result['MACDs_12_26_9']
                    df['macd_hist'] = macd_result['MACDh_12_26_9']
                    df['rsi'] = ta.rsi(df.close)
                    #^ Bollinger Bands
                    bbands_result = ta.bbands(close=df.close)
                    df['lower_bb'] = bbands_result['BBL_5_2.0']
                    df['middle_bb'] = bbands_result['BBM_5_2.0']
                    df['upper_bb'] = bbands_result['BBU_5_2.0']
                    df['bb_width'] = bbands_result['BBB_5_2.0']
                    df['bb_percent'] = bbands_result['BBP_5_2.0']
                    #^ Williams %R
                    df['willr'] = ta.willr(high=df.high, low=df.low, close=df.close)
                    #^ Stochastic Oscillator
                    stoch_result = ta.stoch(high=df.high, low=df.low, close=df.close)
                    df['stoch_k'] = stoch_result['STOCHk_14_3_3']
                    df['stoch_d'] = stoch_result['STOCHd_14_3_3']

                    # Buying and selling signals
                    df['buy_signal'] = ((df['macd_line'] > df['signal_line']) & (df['rsi'] < 30)) | ((df['stoch_k'] > df['stoch_d']) & (df['willr'] < -80))
                    df['sell_signal'] = ((df['macd_line'] < df['signal_line']) & (df['rsi'] > 70)) | ((df['stoch_k'] < df['stoch_d']) & (df['willr'] > -20))

                    signals_df = signals_df.append(df)
                    living_bar()
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
            print(f'We own {len(crypto_positions)} open positions.')
            with alive_bar(len(signals_df)) as living_bar2:
                for index, row in signals_df.iterrows():
                    # update the bar on each iteration with the details of the position
                    living_bar2.text(f'Checking {row["coin"]} | buy: {row["buy_signal"]} | sell: {row["sell_signal"]} |')
                    if row['buy_signal']:
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
                            if position['currency']['code'] == row['coin']:
                                r.order_sell_crypto_limit(symbol=row['coin'],
                                                            quantity=position['quantity'],
                                                            limitPrice=row['close'],
                                                            timeInForce='gtc')
                                self.logger.info(f'Sold {row["coin"]} at {row["close"]}.')
                    living_bar2() # update the bar
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
    def check_stop_loss_prices(self, coins, stop_loss_prices):
        """
        The check_stop_loss_prices function checks if the current price is lower than the stop loss price for any owned coin.
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :doc-author: Trelent
        """
        try:
            for coin in tqdm(coins):
                current_price = float(r.crypto.get_crypto_quote(coin)['mark_price'])
                if current_price < stop_loss_prices[coin]:
                    crypto_positions = r.get_crypto_positions()
                    for position in crypto_positions:
                        if position['currency']['code'] == coin:
                            print(f'Sold {coin} at {current_price} due to stop loss.')
                            r.order_sell_crypto_limit(coin, position['quantity'], current_price)
                            self.logger.info(f'Sold {coin} at {current_price} due to stop loss.')
        except Exception as e:
            self.logger.error(f'Unable to check stop loss prices... {e}')
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
            if utility.is_daytime():
                self.resetter()
                signals_df = self.calculate_ta_indicators(coins)
                self.trading_function(signals_df)
                self.check_stop_loss_prices(coins, stop_loss_prices)
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
        self.utility = Utility() #needed for is_daytime()
        # Set up logging
        self.logger = logging.getLogger('looper')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - %(name)s - %(levelname)s - %(message)s')
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
            utility.log_file_size_checker()
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
    # using ini file to store credentials
    config = configparser.ConfigParser()
    config.read('config/credentials.ini')
    username = config['credentials']['username']
    password = config['credentials']['password']
    #^ create an instance of the Robinhood class
    trader = Trader(username = username, password = password) #^ create an instance of the Trader class
    #^ set stop losses for each coin by multiplying the current price by the stop loss percent (0.05) and subtracting that from the current price (to get the stop loss price).
    stop_loss_prices = {coin: float(r.crypto.get_crypto_quote(coin)['mark_price']) - (float(r.crypto.get_crypto_quote(coin)['mark_price']) * stop_loss_percent) for coin in coins}
    print(f'Stop loss prices: {stop_loss_prices}')
    looper = Looper(trader) #^ create an instance of the Looper class (which will run the Trader class)
    asyncio.run(looper.main_looper(coins, stop_loss_prices)) #^ run the main_looper function