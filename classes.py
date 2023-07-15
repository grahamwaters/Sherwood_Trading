import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime

import pandas as pd
import pandas_ta as ta
from colorama import Back, Fore, Style
from pytz import timezone
from ratelimit import limits, sleep_and_retry
from robin_stocks import robinhood as r
from tqdm import tqdm

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

        # set the values for the available coins, the stop loss percentage (globally), buying power from the file `running_config.json` in the `config` directory
        with open('config/running_config.json', 'r') as f:
            config = json.load(f)
            # if float(config['available_coins'][0]) has nothing in it then the available_coins list is empty so set to 0
            try:
                if config['available_coins'][0] == '' or len(config['available_coins'][0]) == 0 or config['available_coins'][0] == ' ' or config['available_coins'][0] == '[]' or config['available_coins'][0] == '{}':
                    self.available_coins = 0
                else:
                    self.available_coins = float(config['available_coins'][0])
            except Exception as e:
                self.available_coins = 0
                logging.error(f'Unable to get available coins... {e}')
            self.stop_loss_percentage = float(config['stop_loss_percentage'])
            self.buying_power = float(config['buying_power'])
            self.take_profit_percentage = float(config['take_profit_percentage'])
            self.percent_to_use = float(config['play_with_pct']) # how much of the buying power (%) to use while trading

    @sleep_and_retry
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

    @sleep_and_retry
    def resetter(self):
        """
        The resetter function cancels all open orders and sells all positions.
        :doc-author: Trelent
        """
        # check running_config.json to see if the user wants to reset the bot
        # reset_mode = false indicates that the user does not want to reset the bot, and true indicates that the user does want to reset the bot
        with open('config/running_config.json', 'r') as f:
            if json.load(f)['reset_mode'] == 'false':
                return #^ if the user does not want to reset the bot, then return from the function, and don't sell positions or cancel orders
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

    def check_stop_loss_prices(self, coins):
        """
        The check_stop_loss_prices function checks if the current price of a coin is below the stop loss price. If it is, the coin is sold.
        :param coins: A list of coins to check the stop loss price for
        :return: None
        :doc-author: Trelent
        """
        try:
            crypto_positions = r.get_crypto_positions()
            for position in tqdm(crypto_positions):
                if position['currency']['code'] in coins:
                    current_price = float(r.get_crypto_quote(position['currency']['code'], 'mark_price')['mark_price'])
                    stop_loss_price = float(position['cost_bases'][0]['direct_cost_basis']) * (1 - self.stop_loss_percentage)
                    if current_price <= stop_loss_price:
                        r.order_sell_crypto_limit(position['currency']['code'], position['quantity'], position['cost_bases'][0]['direct_cost_basis'])
                        self.logger.info(f'Stop loss triggered for {position["currency"]["code"]}.')
        except Exception as e:
            self.logger.error(f'Unable to check stop loss prices... {e}')

    def update_buying_power(self):
        """
        The update_buying_power function updates the buying power from the Robinhood account and saves it to the `running_config.json` file in the `config` directory.
        :return: None
        :doc-author: Trelent
        """
        try:
            self.buying_power = float(r.load_account_profile()['crypto_buying_power'])
            with open('config/running_config.json', 'r') as f:
                config = json.load(f)
                config['buying_power'] = self.buying_power
            with open('config/running_config.json', 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.logger.error(f'Unable to update buying power... {e}')

    def update_available_coins(self):
        """
        The update_available_coins function updates the available coins from the Robinhood account and saves it to the `running_config.json` file in the `config` directory.
        :return: None
        :doc-author: Trelent
        """
        try:
            self.available_coins = float(r.load_account_profile()['crypto_quantity_available'])
            with open('config/running_config.json', 'r') as f:
                config = json.load(f)
                config['available_coins'] = self.available_coins
            with open('config/running_config.json', 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.logger.error(f'Unable to update available coins... {e}')

    def update_take_profit_percentage(self):
        """
        The update_take_profit_percentage function updates the take profit percentage from the Robinhood account and saves it to the `running_config.json` file in the `config` directory.
        :return: None
        :doc-author: Trelent
        """
        try:
            self.take_profit_percentage = float(r.load_account_profile()['crypto_max_liquidation_price_multiplier'])
            with open('config/running_config.json', 'r') as f:
                config = json.load(f)
                config['take_profit_percentage'] = self.take_profit_percentage
            with open('config/running_config.json', 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.logger.error(f'Unable to update take profit percentage... {e}')

    def update_stop_loss_percentage(self):
        """
        The update_stop_loss_percentage function updates the stop loss percentage from the Robinhood account and saves it to the `running_config.json` file in the `config` directory.
        :return: None
        :doc-author: Trelent
        """
        try:
            self.stop_loss_percentage = float(r.load_account_profile()['crypto_max_liquidation_price_multiplier'])
            with open('config/running_config.json', 'r') as f:
                config = json.load(f)
                config['stop_loss_percentage'] = self.stop_loss_percentage
            with open('config/running_config.json', 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.logger.error(f'Unable to update stop loss percentage... {e}')

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

            coin_lot_signals = {} # this will hold signals for coin lots (batches) that have been bought at different times

            # if the transaction_log.json file exists, load it into the coin_lot_signals dictionary with coin lots as keys and [purchase_price, quantity] as values
            if os.path.exists('config/transaction_log.json'):
                with open('config/transaction_log.json', 'r') as f:
                    transaction_log = json.load(f)
                    for coin in transaction_log:
                        coin_lot_signals[coin] = transaction_log[coin]

            # for each coin in the list of coins to generate signals for

            for coin in coins:
                df = utility.get_last_100_days(coin)
                current_price = float(df.close.iloc[-1])
                df['sma'] = df.close.rolling(window=50).mean()
                df['ema'] = df.close.ewm(span=50, adjust=False).mean()
                df['macd_line'], df['signal_line'], df['macd_hist'] = ta.macd(df.close)
                df['rsi'] = ta.rsi(df.close)
                # df['williams'] = ta.williams_r(df.high, df.low, df.close)
                # df['stochastic_k'], df['stochastic_d'] = ta.stoch(df.high, df.low, df.close)
                # df['bollinger_l'], df['bollinger_m'], df['bollinger_u'] = ta.bollinger_bands(df.close)
                # df['buy_signal'] = ((df.macd_line > df.signal_line) & (df.rsi < 30)) | ((df.stochastic_k > df.stochastic_d) & (df.williams < -80))
                # df['sell_signal'] = ((df.macd_line < df.signal_line) & (df.rsi > 70)) | ((df.stochastic_k < df.stochastic_d) & (df.williams > -20))

                #^ we need to add the parabolic SAR indicator here
                #df['parabolic_sar'] = ta.sar(df.high, df.low, acceleration=0.02, maximum=0.2)
                #^ now we need the stop loss price as of the time of purchase (for this lot of coins)
                df['current_price'] = current_price # this is the current price of the coin
                # purchase price
                df['purchase_price'] = coin_lot_signals[coin][0] # this is the purchase price for this lot of coins
                # stop loss price
                df['stop_loss_price'] = df['purchase_price'] * (1 - self.stop_loss_percentage) # this is the stop loss price for this lot of coins, which is the purchase price * (1 - stop loss percentage)
                # take profit price (take_profit_percentage)
                df['take_profit_price'] = df['purchase_price'] * (1 + self.take_profit_percentage) # this is the take profit price for this lot of coins, which is the purchase price * (1 + take profit percentage)

                signals_df = signals_df.append(df)
            return signals_df

        except Exception as e:
            self.logger.error(f'Unable to generate trading signals... {e}')
            return pd.DataFrame()

    def set_config_values(self, config):
        # this function sets the values in the config file
        # this function is called by the main.py file
        # this function is called when the user wants to change the values in the config file
        # the file is: running_config.json in the config directory
        file = 'config/running_config.json'
        #^ Scenario One: There is an Existing Config File
        if os.path.exists(file):
            # then we load the existing config file
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    existing_config = json.load(f)
                    # now we need to update the existing config file with the new values
                    # we will use the update() method
                    existing_config.update(config)
                    # now we need to save the updated config file
                    with open(file, 'w', encoding='utf-8') as f:
                        json.dump(existing_config, f, indent=4)
                except Exception as e:
                    self.logger.error(f'Unable to set config values... {e}')
        #^ Scenario Two: There is No Existing Config File
        else:
            # then we create a new config file
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)

        #* Now we can log that we have set the config values
        logging.info('Config values have been set. The bot will now restart with the new values.')

    def get_config_values(self):
        # this function gets the values from the config file
        # this function is called by the main.py file
        with open('config/running_config.json', 'r', encoding='utf-8') as f:
            try:
                config = json.load(f)
                return config
            except Exception as e:
                self.logger.error(f'Unable to get config values... {e}')
                return {}

    @sleep_and_retry
    def trading_function(self, signals_df):
        """
        The trading_function function takes the trading signals generated by calculate_ta_indicators() and places trades accordingly.
        :param signals_df: A DataFrame with the trading signals for each coin
        :doc-author: Trelent
        """
        try:
            crypto_positions = r.get_crypto_positions()
            time.sleep(random.randint(1, 3))
            for _, row in signals_df.iterrows():
                coin = row['coin']
                close_price = row['close']

                if row['buy_signal']:
                    Quant = 0
                    with open('config/running_config.json', 'r', encoding='utf-8') as f:
                        try:
                            config = json.load(f)
                            coin_momentums = config.get('coin_momentums', {})
                            momentum = coin_momentums.get(coin, 0)
                            Quant = float(self.buying_power * self.percent_to_use / close_price * (1 / momentum))
                        except Exception as e:
                            print(Fore.RED + f'Unable to load running_config.json... {e}' + Style.RESET_ALL)
                    if Quant == 0:
                        Quant = self.buying_power / close_price * self.percent_to_use

                    block_text = f"""
                    {coin} bought at {close_price} because:
                    - MACD Line: {row['macd_line']}
                    - Signal Line: {row['signal_line']}
                    - RSI: {row['rsi']}
                    - Williams %R: {row['williams']}
                    - Stochastic K: {row['stochastic_k']}
                    - Stochastic D: {row['stochastic_d']}
                    """
                    print(Fore.GREEN + block_text + Style.RESET_ALL)

                    buying_power = self.update_buying_power()
                    if buying_power > 0:
                        r.order_buy_crypto_limit(
                            symbol=coin,
                            quantity=Quant,
                            limitPrice=close_price,
                            timeInForce='gtc'
                        )
                        self.logger.info(f'Bought {coin} at {close_price}.')
                        print(Fore.GREEN + f'Bought {coin} at {close_price}.' + Style.RESET_ALL)
                        with open('config/transaction_log.json', 'r') as f:
                            data = json.load(f)
                            data['transactions'].append({
                                'lot_coin': coin,
                                'lot_purchase_price': close_price,
                                'lot_quantity': buying_power / close_price,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })

                if row['sell_signal'] or row['stop_loss'] or row['take_profit']:
                    Quant = 0
                    for position in crypto_positions:
                        if position['currency']['code'] == coin and float(position['quantity']) > 0:
                            with open('config/transaction_log.json', 'r',encoding='utf-8') as f:
                                data = json.load(f)
                                for transaction in data['transactions']:
                                    if transaction['lot_coin'] == coin and float(transaction['lot_purchase_price']) < close_price:
                                        Quant += float(transaction['lot_quantity'])
                                if len(data['transactions']) == 0:
                                    if float(position['quantity']) > 0:
                                        Quant += float(position['quantity'])

                    if Quant > 0:
                        r.order_sell_crypto_limit(
                            symbol=coin,
                            quantity=Quant,
                            limitPrice=close_price,
                            timeInForce='gtc'
                        )
                        self.logger.info(f'Sold {coin} at {close_price}.')
                        print(Fore.RED + f'Sold {coin} at {close_price}.' + Style.RESET_ALL)
                        with open('config/transaction_log.json', 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            data['transactions'].append({
                                'lot_coin': coin,
                                'lot_purchase_price': close_price,
                                'lot_quantity': Quant,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })

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

class Utility:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

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
            print('Log file size checked and reduced to 1000 lines.')
            await asyncio.sleep(1200) # sleep for 20 minutes

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

    async def check_config(self):
        # check the `running_config.json` file in the `config` directory and return the values for the available coins, the stop loss percentage, and buying power, as well as the percent_to_use and stop_loss_percentage which are generated in the Trader Class when it is initialized. They can be found in the config file though, at any time.
        with open('config/running_config.json', 'r') as f:
            config = json.load(f)
            return config['available_coins'], config['stop_loss_percentage'], config['buying_power'], config['percent_to_use']

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

    def set_username(self, username):
        """
        The set_username function sets the username for the Trader class.
        :param username: The username to set
        :doc-author: Trelent
        """
        self.trader.username = username # set the username for the Trader class

    def set_password(self, password):
        """
        The set_password function sets the password for the Trader class.
        :param password: The password to set
        :doc-author: Trelent
        """
        self.trader.password = password

    def set_buying_power(self, buying_power):
        """
        The set_buying_power function sets the buying power for the Trader class.
        :param buying_power: The buying power to set
        :doc-author: Trelent
        """
        self.trader.buying_power = buying_power

    def set_stop_loss_percentage(self, stop_loss_percentage):
        """
        The set_stop_loss_percentage function sets the stop loss percentage for the Trader class.
        :param stop_loss_percentage: The stop loss percentage to set
        :doc-author: Trelent
        """
        self.trader.stop_loss_percentage = stop_loss_percentage
