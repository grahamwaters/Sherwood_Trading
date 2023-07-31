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
import time
import os
import random
# from legacy.V5.main2 import stop_loss_percent
from tqdm import tqdm

PCT_SPEND = 0.05 # 5% of buying power
verbose_mode = False # Set to True to see more logging output

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
    async def log_file_size_checker(self):
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
            return True #! Note: Change back to False to prevent trading during the night
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
        self.max_investment_per_coin = 0.01 # the max amount to invest in each coin
        self.logger = logging.getLogger('trader')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'{Fore.YELLOW}%(asctime)s{Style.RESET_ALL} - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # get the spend pct from the config file
        self.spend_pct = config['trader']['spend_pct']
        # get the stop loss pct from the config file
        self.stop_loss_pct = config['trader']['stop_loss_pct']
        # Add the exception hook to log uncaught exceptions
        sys.excepthook = self.handle_uncaught_exception

        self.utility = Utility() # create an instance of the Utility class
        self.looper = Looper(
            self
            ) # create an instance of the Looper class

        self.signals_df = pd.DataFrame() # create an empty DataFrame to store the signals
        # Login to Robinhood
        self.login_setup()
        # set up coin inertia dictionary
        self.coin_inertia = {} # this is to slow down the buying and selling of a coin to prevent rapid buying and selling of a coin. Defaults to 1 and increments by 1 every time a coin is bought. Then every ten minutes it decrements by 1 until it reaches 1 again.
        # set up the inertias for the coins that are already owned
        for coin in r.get_crypto_positions():
            self.coin_inertia[coin['currency']['code']] = 1
        # set up the coin inertia timer
        self.coin_inertia_timer = 0
        # set up the coin inertia timer max
        self.coin_inertia_timer_max = 10
        # set up the coin inertia timer increment
        self.coin_inertia_timer_increment = 1
        # set up the coin inertia timer decrement
        self.coin_inertia_timer_decrement = 1
        # set up the coin inertia timer reset
        self.coin_inertia_timer_reset = 0

    def handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """
        Custom exception hook to log uncaught exceptions with traceback.
        """
        self.logger.error("Uncaught exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))
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

                    #^ coins name
                    df['coin'] = coin

                    # Buying and selling signals
                    df['buy_signal'] = ((df['macd_line'] > df['signal_line']) & (df['rsi'] < 30)) | ((df['stoch_k'] > df['stoch_d']) & (df['willr'] < -80))
                    df['sell_signal'] = ((df['macd_line'] < df['signal_line']) & (df['rsi'] > 70)) | ((df['stoch_k'] < df['stoch_d']) & (df['willr'] > -20))

                    if verbose_mode:
                        # if a buy signal is generated, print the buy signal as a line
                        if df['buy_signal'].iloc[-1]:
                            print(Fore.GREEN + f'Buy signal generated for {coin}.' + Style.RESET_ALL)
                        # if a sell signal is generated, print the sell signal as a line
                        if df['sell_signal'].iloc[-1]:
                            print(Fore.RED + f'Sell signal generated for {coin}.' + Style.RESET_ALL)
                    signals_df = signals_df.append(df)
                    living_bar()
                return signals_df
            except Exception as e:
                self.logger.error(f'Unable to generate trading signals... {e}')
                return pd.DataFrame()

    def update_master_order_history(self, coin, close_price, spend_amount, side):
        # update Master Order History with new order
        if side == 'buy':
            # then spend_amount is negative and represents the amount of money spent in USD
            spend_amount = spend_amount * -1
            volume_sold = 0
        elif side == 'sell':
            # then spend_amount is the volume of the coin sold
            volume_sold = spend_amount
            spend_amount = 0
        else:
            self.logger.error('Unable to update master order history... Invalid side.')
            return
        try:
            master_order_history = pd.read_csv('data/master_order_list.csv')
            master_order_history = master_order_history.append({'coin': coin, 'close_price': close_price, 'spend_amount': spend_amount, 'volume': volume_sold, 'date': datetime.now(), 'side': side}, ignore_index=True)
            master_order_history.to_csv('data/master_order_list.csv', index=False)
        except Exception as e:
            self.logger.error(f'Unable to update master order history... {e}')

    def update_coin_inertia(self, coin):
        # update the coin inertia by incrementing it by 1
        try:
            coin_inertia = pd.read_csv('data/coin_inertia.csv')
            coin_inertia.loc[coin_inertia['coin'] == coin, 'inertia'] += 1
            coin_inertia.to_csv('data/coin_inertia.csv', index=False)
        except Exception as e:
            self.logger.error(f'Unable to update coin inertia... {e}')

    def check_coin_inertia(self, coin):
        # check the coin inertia
        try:
            coin_inertia = pd.read_csv('data/coin_inertia.csv')
            return coin_inertia.loc[coin_inertia['coin'] == coin, 'inertia'].iloc[0]
        except Exception as e:
            self.logger.error(f'Unable to check coin inertia... {e}')
            return 0

    def ordering_function(self, coin, spend_amount,side, df):
        """
        The ordering_function function takes in a coin, spend_amount, side and dataframe as arguments.
        The function then determines the side of the order (buy or sell) and places an order for that amount.
        If we are selling then we need to sell ALL of the coin that we have.

        :param coin: Determine which coin to buy or sell
        :param spend_amount: Determine how much money we want to spend on the order
        :param side: Determine whether to buy or sell
        :param df: Pass in the dataframe that is created by the get_data function
        :return: A dictionary of the order information
        :doc-author: Trelent
        """
        try:
            coins_held = r.get_crypto_positions(info='quantity_available', symbol=coin, jsonify=True)
            coins_held = coins_held[0]['quantity_available']
            # determine the side of the order
            # side = 'buy' or 'sell'
            amount_in = 'dollars' if side == 'buy' else 'amount'
            quantity_or_price = spend_amount / df['close'] if side == 'buy' else float(coins_held)

            quantity_or_price = str(spend_amount / df['close'].iloc[-1]) if side == 'buy' else str(df['quantity_available'].iloc[-1])
            # place the buy order
            r.orders.order_crypto(
                symbol=coin,
                quantityOrPrice=float(quantity_or_price),
                amountIn=str(amount_in),
                side=side,
                timeInForce='gtc',
                jsonify=True
                )

            if side == 'buy':
                print(f'Buying {quantity_or_price} {coin} at {df["close"].iloc[-1]}')
            else:
                print(f'Selling {quantity_or_price} {coin} at {df["close"].iloc[-1]}')
        except Exception as e:
            print(e)

    def trading_function(self, signals_df):
        """
        The trading_function function takes the trading signals generated by calculate_ta_indicators() and places trades accordingly.
        :param signals_df: A DataFrame with the trading signals for each coin
        :doc-author: Trelent
        """
        try:
            buying_power = self.update_buying_power()
            crypto_positions = r.get_crypto_positions()

            for coin in signals_df['coin'].unique():
                # Get the last row of the signals_df for the current coin
                df = signals_df.loc[signals_df['coin'] == coin].iloc[-1]

                # Check if the last row of the signals_df is a buy signal and we have enough buying power to buy the coin
                if df['buy_signal'] and buying_power > df['close']:
                    # Buy the coin
                    quantity_to_buy = min(buying_power, self.max_investment_per_coin)
                    self.ordering_function(coin, quantity_to_buy, 'buy', df)
                    # Update the master order history
                    self.update_master_order_history(coin, df['close'], quantity_to_buy, 'buy')
                elif df['sell_signal']:
                    # Sell the coin
                    self.ordering_function(coin, 0, 'sell', df)
                    # Update the master order history
                    self.update_master_order_history(coin, df['close'], df['quantity_available'], 'sell')
                else:
                    # Do nothing
                    pass

        except Exception as e:
            self.logger.error(f'Unable to execute trading function... {e}')




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
            # buying_power = cash_available + crypto_dollars
            # remember to load the config file and get the pct to use for buying power
            # load the config file
            config = configparser.ConfigParser()
            config.read('config/credentials.ini')
            # get the pct to use for buying power
            spend_pct = float(config['trading']['percent_to_use'])

            return cash_available * spend_pct # this is the percent of buying power to use
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

                # signals_df = self.calculate_ta_indicators(coins)
                self.trading_function(signals_df)
                self.check_stop_loss_prices(coins, stop_loss_prices)
            else:
                self.logger.info('It is not daytime. The main function will not run.')
                # reduce the spend by 1/2 if it's not daytime
                self.spend = self.spend / 2
                #todo: gpt needs to complete this at some point.
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
                # # update the inertia values every ten minutes by decrementing them unless they're already at one
                # for coin in coins:
                #     if self.trader.inertia_values[coin] > 1:
                #         # save the inertia values to a file
                #         with open('inertia_values.json', 'w') as f:

            self.trader.main(coins, stop_loss_prices)
            # run all async functions simultaneously
            # log_file_size_checker included to prevent log file from getting too large
            await self.utility.log_file_size_checker()
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
                # use alive progress to show a sleep bar for ten minutes
                with alive_bar(600, bar='blocks', spinner='dots_waves') as bar:
                    for i in range(600):
                        bar()
                        await asyncio.sleep(1)
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