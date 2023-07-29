import configparser
import threading
import os
from datetime import datetime
import logging
import pandas_ta as ta
from robin_stocks import robinhood as r
from colorama import Fore, Style
from tqdm import tqdm
import asyncio
import time
import os
import time
import json
import numpy as np
import logging
import pandas as pd
import re
import time
from alive_progress import alive_bar
import threading
from main_sock import verbose_mode
# import timezone
from pytz import timezone
from icecream import ic
# import pandas ta library

class Batch:
    def __init__(self, coin, purchase_price, quantity):
        ic()
        self.verbose_mode_batch = False
        self.coin = coin
        self.purchase_price = purchase_price
        self.quantity = quantity
        self.logger = logging.getLogger('batch')
        if self.verbose_mode_batch:
            print(f'Batch created for {self.coin} at {self.purchase_price} with {self.quantity} quantity')
        else:
            pass

    def get_stop_loss_price(self, stop_loss_percent):
        return self.purchase_price * (1 - stop_loss_percent)


class Utility:

    def __init__(self, logger):
        # Initialize the logger
        ic()
        self.logger = logger


    def log_file_size_checker(self):
        """
        The log_file_size_checker function checks the size of the log file.
        If it exceeds 1000 lines, it will remove all but the last 1000 lines.

        :param self: Represent the instance of the object itself
        :return: None
        """
        log_file = 'logs/robinhood.log'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1000:
                    num_lines_to_remove = len(lines) - 1000
                    with open(log_file, 'w') as f:
                        f.writelines(lines[num_lines_to_remove:])

    def get_last_100_days(self, coin):

        # strip the coin of any non-alphanumeric characters
        coin = re.sub(r'\W+', '', coin)

        try:
            df = pd.DataFrame(r.crypto.get_crypto_historicals(coin, interval='hour', span='3month', bounds='24_7'))
            df = df.set_index('begins_at')
            df.index = pd.to_datetime(df.index)
            df = df.loc[:, ['close_price', 'open_price', 'high_price', 'low_price']]
            df = df.rename(columns={'close_price': 'close', 'open_price': 'open', 'high_price': 'high', 'low_price': 'low'})
            df = df.apply(pd.to_numeric)
            return df
        except Exception as e:
            self.logger.error(f"Unable to get historical data for {coin}. Error: {e}")
            return None


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



class Trader:
    def __init__(self, logger, utility_logger):
        ic()
        self.logger = logging.getLogger('trader')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.load_config() #^ this is the function that loads the config file and populates the variables
        self.login_setup(self.username, self.password)

        self.utility = Utility(self.logger)
        self.batches = []  # Initialize batches attribute here


    def load_config(self):
        # Load config file
        config_file = 'config/credentials.ini'
        if os.path.exists(config_file):
            # Load config file
            config = configparser.ConfigParser()
            config.read(config_file)
            self.username = config['credentials']['username']
            self.password = config['credentials']['password']
            self.coins = config['trading']['coins'].split(',')
            self.stop_loss_percent = float(config['trading']['stop_loss_percent'])
            self.verbose_mode = config.getboolean('logging', 'verbose_mode')
            self.debug_verbose = config.getboolean('logging', 'debug_verbose')
            self.percent_to_use = float(config['trading']['percent_to_use'])
            self.logger.info(f'Loaded config file with username: {self.username}, password: xxxxxx, coins: {self.coins}, stop_loss_percent: {self.stop_loss_percent}, verbose_mode: {self.verbose_mode}, debug_verbose: {self.debug_verbose}, percent_to_use: {self.percent_to_use}')
        else:
            self.logger.error('Unable to load config file.')
            exit(1)

    def login_setup(self, username, password):
        try:
            r.login(self.username, self.password)
            time.sleep(1)
            self.logger.info('Logged in to Robinhood successfully.')
        except Exception as e:
            self.logger.error(f'Unable to login to Robinhood... {e}')
            # try again in 5 seconds
            time.sleep(5)
            try:
                self.login_setup(self.username, self.password)
            except Exception as e:
                self.logger.error(f'Unable to login to Robinhood... {e}')
                exit(1)

    def load_config(self):
        config = configparser.ConfigParser()
        config.read('config/credentials.ini')
        self.username = config['credentials']['username']
        self.password = config['credentials']['password']
        self.coins = config['trading']['coins'].split(',')
        self.stop_loss_percent = float(config['trading']['stop_loss_percent'])
        self.verbose_mode = config.getboolean('logging', 'verbose_mode')
        self.debug_verbose = config.getboolean('logging', 'debug_verbose')
        self.percent_to_use = float(config['trading']['percent_to_use'])
        print(f'Loaded config file with username: {self.username}, password: xxxxxx, coins: {self.coins}, stop_loss_percent: {self.stop_loss_percent}, verbose_mode: {self.verbose_mode}, debug_verbose: {self.debug_verbose}, percent_to_use: {self.percent_to_use}')

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

    def trading_function_old(self, signals_df):
        try:
            # retrieve the percent_to_use from the ini file
            percent_to_use = self.percent_to_use
            # buying power
            buying_power = percent_to_use * float(r.load_account_profile()['buying_power'])
            # retrieve the stop_loss_percent from the ini file
            stop_loss_percent = self.stop_loss_percent
            crypto_positions = {position['currency']['code']: position for position in r.get_crypto_positions()}

            signals_df['buy_signal'] = signals_df.apply(lambda row: row['indicators'].buy_signal and row['coin'] not in crypto_positions, axis=1)

            signals_df['sell_signal'] = signals_df.apply(lambda row: row['indicators'].sell_signal and row['coin'] in crypto_positions, axis=1)


            with alive_bar(len(signals_df), bar='blocks', spinner='dots_waves') as bar:
                for _, row in signals_df.iterrows():  # Correct indentation
                    coin = row['coin']
                    close_price = row['close']
                    indicators = row['indicators']
                    if self.verbose_mode:
                        print(f'-- {coin} -- + {indicators["buy_signal"]} - {indicators["sell_signal"]}') #*  cosmetic

                    if row['buy_signal']:
                        if self.verbose_mode:
                            self.logger.info(Fore.GREEN + f'Bought {coin} at {close_price} because:\n' + '\n'.join(f'- {key}: {value}' for key, value in indicators.items()) + Style.RESET_ALL) #*  cosmetic
                        else:
                            self.logger.info(f'Bought {coin} at ${close_price}') #*  cosmetic
                        buying_power = self.update_buying_power()
                        buying_power = buying_power * percent_to_use
                        if buying_power > 0:
                            quantity = buying_power / close_price
                            order = r.order_buy_crypto_limit(symbol=coin, quantity=quantity, limitPrice=close_price, timeInForce='gtc')
                            if order['state'] == 'filled':
                                self.batches.append(Batch(coin, close_price, quantity))
                    if row['sell_signal'] and coin in crypto_positions:
                        for batch in self.batches:
                            bar.text(f'Selling {batch.coin}...')
                            if batch.coin == coin:
                                current_price = close_price  # Update this with the real-time price if available
                                stop_loss_price = batch.get_stop_loss_price(stop_loss_percent)
                                profit = current_price - batch.purchase_price
                                if current_price < stop_loss_price or profit > 0:
                                    order = r.order_sell_crypto_limit(symbol=coin, quantity=batch.quantity, limitPrice=current_price, timeInForce='gtc')
                                    if order['state'] == 'filled':
                                        self.batches.remove(batch)
                                        if self.verbose_mode:
                                            if profit > 0:
                                                self.logger.info(Fore.GREEN + f'Sold {coin} at {current_price} for a profit of {profit}' + Style.RESET_ALL)
                                            elif profit < 0:
                                                self.logger.info(Fore.RED + f'Sold {coin} at {current_price} for a loss of {profit}' + Style.RESET_ALL)
                                        else:
                                            self.logger.info(f'Sold {coin} at {current_price}')
                                elif self.verbose_mode:
                                    self.logger.info(f'Stop loss for {coin} is {stop_loss_price}. Current price is {current_price}.')
                    bar()
        except Exception as e:
            self.logger.error(f'Unable to trade... {e}')

    def trading_function(self, signals_df):
        try:
            # retrieve the percent_to_use from the ini file
            percent_to_use = self.percent_to_use
            # buying power
            buying_power = percent_to_use * float(r.load_account_profile()['buying_power'])
            # retrieve the stop_loss_percent from the ini file
            stop_loss_percent = self.stop_loss_percent
            crypto_positions = {position['currency']['code']: position for position in r.get_crypto_positions()}

            signals_df['buy_signal'] = signals_df.apply(lambda row: row['indicators'].get('buy_signal', False) and row['coin'] not in crypto_positions, axis=1)
            signals_df['sell_signal'] = signals_df.apply(lambda row: row['indicators'].get('sell_signal', False) and row['coin'] in crypto_positions, axis=1)

            with alive_bar(len(signals_df), bar='blocks', spinner='dots_waves') as bar:
                for _, row in signals_df.iterrows():  # Correct indentation
                    coin = row['coin']
                    close_price = row['close']
                    indicators = row['indicators']
                    if self.verbose_mode:
                        print(f'-- {coin} -- + {indicators["buy_signal"]} - {indicators["sell_signal"]}') #*  cosmetic

                    if row['buy_signal']:
                        if self.verbose_mode:
                            self.logger.info(Fore.GREEN + f'Bought {coin} at {close_price} because:\n' + '\n'.join(f'- {key}: {value}' for key, value in indicators.items()) + Style.RESET_ALL) #*  cosmetic
                        else:
                            self.logger.info(f'Bought {coin} at ${close_price}') #*  cosmetic
                        buying_power = self.update_buying_power()
                        buying_power = buying_power * percent_to_use
                        if buying_power > 0:
                            quantity = buying_power / close_price
                            order = r.order_buy_crypto_limit(symbol=coin, quantity=quantity, limitPrice=close_price, timeInForce='gtc')
                            if order['state'] == 'filled':
                                self.batches.append(Batch(coin, close_price, quantity))
                    if row['sell_signal'] and coin in crypto_positions:
                        for batch in self.batches:
                            bar.text(f'Selling {batch.coin}...')
                            if batch.coin == coin:
                                current_price = close_price  # Update this with the real-time price if available
                                stop_loss_price = batch.get_stop_loss_price(stop_loss_percent)
                                profit = current_price - batch.purchase_price
                                if current_price < stop_loss_price or profit > 0:
                                    order = r.order_sell_crypto_limit(symbol=coin, quantity=batch.quantity, limitPrice=current_price, timeInForce='gtc')
                                    if order['state'] == 'filled':
                                        self.batches.remove(batch)
                                        if self.verbose_mode:
                                            if profit > 0:
                                                self.logger.info(Fore.GREEN + f'Sold {coin} at {current_price} for a profit of {profit}' + Style.RESET_ALL)
                                            elif profit < 0:
                                                self.logger.info(Fore.RED + f'Sold {coin} at {current_price} for a loss of {profit}' + Style.RESET_ALL)
                                        else:
                                            self.logger.info(f'Sold {coin} at {current_price}')
                                elif self.verbose_mode:
                                    self.logger.info(f'Stop loss for {coin} is {stop_loss_price}. Current price is {current_price}.')
                    bar()
        except Exception as e:
            self.logger.error(f'Unable to trade... {e}')


    def calculate_ta_indicators(self, coins):
        try:
            signals = [] # Create a list to store the buy signals and sell signals
            for coin in tqdm(coins):
                print(f'Calculating indicators for {coin}...')
                if coin == 'None' or coin == None:
                    # drop the coin from the list if it is None
                    coins.remove(coin)
                    continue # Skip this iteration if the coin is None
                df = self.utility.get_last_100_days(coin)

                df['sma'] = df.close.rolling(window=50).mean()
                df['ema'] = df.close.ewm(span=50, adjust=False).mean()
                macd_values = ta.macd(df.close)
                df['macd_line'], df['signal_line'], df['macd_hist'] = macd_values['MACD_12_26_9'], macd_values['MACDs_12_26_9'], macd_values['MACDh_12_26_9']
                macd_line, signal_line, macd_hist = df['macd_line'].iat[-1], df['signal_line'].iat[-1], df['macd_hist'].iat[-1]
                df['rsi'] = ta.rsi(df.close)
                # extract the rsi value from the last row of the dataframe
                rsi = df['rsi'].iat[-1]
                df['buy_signal'] = ((macd_line > signal_line) & (rsi < 30)) #^ this indicates that the macd line has crossed above the signal line, and the rsi is below 30 --> buy
                # extract the macd_line value from the last row of the dataframe
                macd_line = df['macd_line'].iat[-1]
                # extract the signal_line value from the last row of the dataframe
                signal_line = df['signal_line'].iat[-1]

                # extract the close price from the last row of the dataframe
                close_price = df['close'].iat[-1]
                # sell signal
                sell_signal = ((macd_line < signal_line) & (rsi > 70)) #^ this indicates that the macd line has crossed below the signal line, and the rsi is above 70 --> sell
                buy_signal = ((macd_line > signal_line) & (rsi < 30)) #^ this indicates that the macd line has crossed above the signal line, and the rsi is below 30 --> buy

                signals.append({
                    'coin': coin,
                    'buy_signal': df['buy_signal'].iat[-1],
                    'sell_signal': df['sell_signal'].iat[-1],
                    'close': df['close'].iat[-1],
                    'indicators': {**df.iloc[-1].to_dict(), 'buy_signal': df['buy_signal'].iat[-1], 'sell_signal': df['sell_signal'].iat[-1]}
                })
                # if we either have a buy or sell signal print the coin and the signal
                if buy_signal or sell_signal:
                    self.logger.info(f'{coin} - Buy: {buy_signal} Sell: {sell_signal}')
            return pd.DataFrame(signals)
        except Exception as e:
            self.logger.error(f'Unable to generate trading signals... {e}')
            return pd.DataFrame()

    def check_stop_loss_prices(self):
        try:
            with alive_bar(len(self.batches), bar='blocks', spinner='dots_waves') as bar:
                for batch in self.batches:
                    current_price = float(r.crypto.get_crypto_quote(batch.coin)['mark_price'])
                    if batch.check_stop_loss(current_price):
                        r.order_sell_crypto_limit(batch.coin, batch.quantity, current_price)
                        self.logger.info(f'Sold {batch.coin} at {current_price} due to stop loss.')
                        self.batches.remove(batch)
                        self.logger(Fore.RED + f'\tSubmitting Sell Order for {batch.quantity} {batch.coin} at {current_price} for a total of ${batch.quantity * current_price}' + Style.RESET_ALL) #*  cosmetic
                        bar()
                    else:
                        bar()
        except Exception as e:
            self.logger.error(f'Unable to check stop loss prices... {e}') #*  cosmetic

    def get_total_crypto_dollars(self):
        total_value = 0.0
        crypto_positions = r.crypto.get_crypto_positions()
        for position in tqdm(crypto_positions):
            quantity = float(position['quantity'])
            coin = position['currency']['code']
            current_price = float(r.crypto.get_crypto_quote(coin)['mark_price'])
            total_value += quantity * current_price
        return total_value

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
            return buying_power * self.percent_to_use
        except Exception as e:
            self.logger.error(f'Unable to update buying power... {e}')
            return 0

    def main_old(self):
        # login to robinhood
        r.login(username=self.username, password=self.password, expiresIn=86400, by_sms=True)
        reset_positions = False
        # load config
        self.load_config()
        if reset_positions:
            self.resetter()
        utility = Utility(logger=self.logger)

        #^ start log_file_size_checker in a separate thread
        threading.Thread(target=utility.log_file_size_checker, daemon=True).start()

        while True:
            # refresh our loaded config to catch any changes
            self.load_config()
            if self.debug_verbose:
                self.logger.setLevel(logging.DEBUG)
            elif self.verbose_mode:
                self.logger.setLevel(logging.INFO)
            else:
                self.logger.setLevel(logging.WARNING)

            #^ first print how much we have in buying power, and how much we have in crypto (by value)
            buying_power = self.update_buying_power() #^ this is the amount of buying power we have
            self.logger.info(f'Buying power: ${buying_power:.2f}')
            crypto_value = self.get_total_crypto_dollars() #^ this is the value of all crypto in dollars
            self.logger.info(f'Crypto value: ${crypto_value:.2f}')
            print(Fore.GREEN + f'Buying power: ${buying_power:.2f}' + Style.RESET_ALL) #*  cosmetic
            print(Fore.GREEN + f'Crypto value: ${crypto_value:.2f}' + Style.RESET_ALL) #*  cosmetic
            print(Fore.GREEN + f'Total value: ${buying_power + crypto_value:.2f}' + Style.RESET_ALL) #*  cosmetic
            print('-'*20) #*  cosmetic
            signals_df = self.calculate_ta_indicators(self.coins) # Fetch the trading signals
            self.trading_function(signals_df) # Pass the signals to the trading function
            self.check_stop_loss_prices()
            # now use log_file_size_checker to check the size of the log file. If it is too big, then we should reset it to 1000 lines (from the end)
            utility.log_file_size_checker() #*  cosmetic
            print('-'*20) #*  cosmetic
            print('sleeping for 60 seconds...') #*  cosmetic
            time.sleep(60)

    def main_old2(self):
        # login to robinhood
        r.login(username=self.username, password=self.password, expiresIn=86400, by_sms=True)
        reset_positions = False
        # load config
        self.load_config()
        if reset_positions:
            self.resetter()
        utility = Utility(logger=self.logger)

        Thread = threading.Thread
        #^ start log_file_size_checker in a separate thread
        threading.Thread(target=utility.log_file_size_checker, daemon=True).start()

        while True:
            # refresh our loaded config to catch any changes
            self.load_config()
            if self.debug_verbose:
                self.logger.setLevel(logging.DEBUG)
            elif self.verbose_mode:
                self.logger.setLevel(logging.INFO)
            else:
                self.logger.setLevel(logging.WARNING)

            threads = []

            # Create new threads
            thread1 = Thread(target=self.calculate_ta_indicators, args=(self.coins,))  # Fetch the trading signals
            thread2 = Thread(target=self.trading_function, args=(self.calculate_ta_indicators(self.coins),))  # Pass the signals to the trading function
            thread3 = Thread(target=self.check_stop_loss_prices)

            # Start new Threads
            # run thread 1 every 5 minutes
            thread1.start() # Fetch the trading signals
            print('thread1 started')
            # run thread 2 every ten minutes
            thread2.start()
            # run thread 3 every 5 minutes (this thread checks the stop loss prices)
            thread3.start()
            print('thread3 started')

            # Add threads to thread list
            threads.append(thread1)
            threads.append(thread2)
            threads.append(thread3)

            print(f'waiting for threads to complete...')
            # Wait for all threads to complete
            for t in threads:
                t.join()

            # now use log_file_size_checker to check the size of the log file. If it is too big, then we should reset it to 1000 lines (from the end)
            utility.log_file_size_checker() #*  cosmetic
            print('-'*20) #*  cosmetic
            print('sleeping for 60 seconds...') #*  cosmetic
            time.sleep(60)



    def main(self):
        def run_thread1():
            # with alive_bar(300, bar='blocks', spinner='dots_waves') as bar:
            #     for i in range(300):
            #         time.sleep(1)
            #         bar()
            self.calculate_ta_indicators(self.coins) # Fetch the trading signals
            # Schedule the next run in 5 minutes (300 seconds)
            threading.Timer(300, run_thread1).start()

        def run_thread2():
            # with alive_bar(600, bar='blocks', spinner='dots_waves') as bar:
            #     for i in range(600):
            #         time.sleep(1)
            #         bar()
            signals_df = self.calculate_ta_indicators(self.coins) # Fetch the trading signals
            self.trading_function(signals_df) # Pass the signals to the trading function
            # Schedule the next run in 10 minutes (600 seconds)
            threading.Timer(600, run_thread2).start()

        def run_thread3():
            # with alive_bar(300, bar='blocks', spinner='dots_waves') as bar:
            #     for i in range(300):
            #         time.sleep(1)
            #         bar()
            self.check_stop_loss_prices()
            # Schedule the next run in 5 minutes (300 seconds)
            threading.Timer(300, run_thread3).start()

        # Start the threads
        run_thread1()
        run_thread2()
        run_thread3()


# use the new main
if __name__ == '__main__':
    # instantiate the trader class and run the main function using the config file credentials.ini

    #& This is the logger for the trader class
    logger = logging.getLogger('trader') #^  instantiate the logger
    logger.setLevel(logging.INFO) #^  set the logger level
    handler = logging.FileHandler('trader.log') #^  instantiate the file handler
    handler.setLevel(logging.INFO) #^  set the file handler level
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #^  instantiate the formatter
    handler.setFormatter(formatter) #^  set the formatter
    logger.addHandler(handler) #^  add the handler to the logger

    #& This is the logger for the utility class
    utility_logger = logging.getLogger('utility') #^  instantiate the logger
    utility_logger.setLevel(logging.INFO) #^  set the logger level
    utility_handler = logging.FileHandler('utility.log') #^  instantiate the file handler
    utility_handler.setLevel(logging.INFO) #^  set the file handler level
    utility_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #^  instantiate the formatter
    utility_handler.setFormatter(utility_formatter) #^  set the formatter
    utility_logger.addHandler(utility_handler) #^  add the handler to the logger

    trader = Trader(logger=logger, utility_logger=utility_logger)
    trader.main()
