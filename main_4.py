
import logging
import pandas as pd
import pandas_ta as ta
import robin_stocks as rstocks
from robin_stocks import robinhood as r
from datetime import datetime
from pytz import timezone
import asyncio
from colorama import Fore, Back, Style
import json
import os

class Utility:
    def __init__(self):
        pass

    async def log_file_size_checker(self):
        """
        The log_file_size_checker function is a coroutine that runs in the background of the program.
        It checks to see if there are more than 1000 lines in the log file, and if so, it removes all but
        the last 1000 lines from the log file.

        :param self: Represent the instance of the class
        :return: A coroutine object
        :doc-author: Trelent
        """
        while True:
            with open('logs/robinhood.log', 'r') as f:
                lines = f.readlines()
                if len(lines) > 1000:
                    num_lines_to_remove = len(lines) - 1000
                    with open('logs/robinhood.log', 'w') as f:
                        f.writelines(lines[num_lines_to_remove:])
            await asyncio.sleep(1200)

    async def get_last_100_days(self, coin):
        """
        The get_last_100_days function takes in a coin and returns the last 100 days of data for that coin.
            The function uses the robin_stocks library to get historical data from Robinhood's Crypto API.
            It then sets the index as 'begins_at' which is a datetime object, and renames columns to match
            what pandas-ta expects.

        :param self: Allow the function to access other functions within the class
        :param coin: Specify which coin you want to get data for
        :return: A dataframe with the following columns:
        :doc-author: Trelent
        """

        try:
            df = pd.DataFrame(await r.crypto.get_crypto_historicals(coin, interval='hour', span='3month', bounds='24_7'))
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
        The is_daytime function checks the current time and returns True if it is between 8am and 8pm,
            otherwise it returns False. This function is used to determine whether or not to run the
            trading algorithm.

        :param self: Represent the instance of the object itself
        :return: True if the current hour is between 8 and 20, otherwise it returns false
        :doc-author: Trelent
        """

        current_time = datetime.now(timezone('US/Central'))
        current_hour = current_time.hour
        if 8 <= current_hour <= 20:
            return True
        else:
            return False

class Trader:
    def __init__(self, username, password):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the logger and login credentials for Robinhood.

        :param self: Represent the instance of the class
        :param username: Set the username attribute of the class
        :param password: Set the password for the user
        :return: An object that is an instance of the class
        :doc-author: Trelent
        """

        self.username = username
        self.password = password
        self.logger = logging.getLogger('trader')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.login_setup()

    def login_setup(self):
        """
        The login_setup function is used to login to Robinhood.
            It takes in the username and password as parameters, and logs into Robinhood using those credentials.
            If it fails, it will log an error message.

        :param self: Refer to the current instance of the class
        :return: A boolean value: true or false
        :doc-author: Trelent
        """

        try:
            r.login(self.username, self.password)
            self.logger.info('Logged in to Robinhood successfully.')
        except Exception as e:
            self.logger.error(f'Unable to login to Robinhood... {e}')

    async def resetter(self):
        """
        The resetter function is used to cancel all open orders and sell all positions.
        It is called when the bot encounters an error that it cannot recover from, or if the user manually calls it.

        :param self: Represent the instance of the class
        :return: A list of all orders and positions that were cancelled
        :doc-author: Trelent
        """

        try:
            open_orders = await r.get_all_open_crypto_orders()
            print(Fore.YELLOW + 'Canceling all open orders...' + Style.RESET_ALL)
            for order in open_orders:
                await r.cancel_crypto_order(order['id'])
            print(Fore.GREEN + 'All open orders cancelled.')
            self.logger.info('All open orders cancelled.' + Style.RESET_ALL)
            crypto_positions = await r.get_crypto_positions()
            for position in crypto_positions:
                await r.order_sell_crypto_limit(position['currency']['code'], position['quantity'], position['cost_bases'][0]['direct_cost_basis'])
            self.logger.info('All positions sold.')
        except Exception as e:
            self.logger.error(f'Unable to reset orders and positions... {e}')

    async def calculate_ta_indicators(self, coins):
        """
        The calculate_ta_indicators function is the main function that will be called by the bot.
        It should return a dataframe with buy and sell signals for each coin in coins.
        The columns of this dataframe should be:
            - 'buy_signal' (boolean) indicating whether or not to buy a coin at current price,
            - 'sell_signal' (boolean) indicating whether or not to sell a coin at current price, and
            - any other indicators you want displayed on your dashboard.

        :param self: Represent the instance of the class
        :param coins: Specify which coins you want to get trading signals for
        :return: A dataframe of trading signals
        :doc-author: Trelent
        """

        try:
            utility = Utility()
            signals_df = pd.DataFrame()
            for coin in coins:
                df = await utility.get_last_100_days(coin)
                df['sma'] = df.close.rolling(window=50).mean()
                df['ema'] = df.close.ewm(span=50, adjust=False).mean()
                df['macd_line'], df['signal_line'], df['macd_hist'] = ta.macd(df.close)
                df['rsi'] = ta.rsi(df.close)
                df['williams'] = ta.williams_r(df.high, df.low, df.close)
                df['stochastic_k'], df['stochastic_d'] = ta.stoch(df.high, df.low, df.close)
                df['bollinger_l'], df['bollinger_m'], df['bollinger_u'] = ta.bollinger_bands(df.close)
                df['buy_signal'] = ((df.macd_line > df.signal_line) & (df.rsi < 30)) | ((df.stochastic_k > df.stochastic_d) & (df.williams < -80))
                df['sell_signal'] = ((df.macd_line < df.signal_line) & (df.rsi > 70)) | ((df.stochastic_k < df.stochastic_d) & (df.williams > -20))
                signals_df = signals_df.append(df)
            return signals_df
        except Exception as e:
            self.logger.error(f'Unable to generate trading signals... {e}')
            return pd.DataFrame()

    async def trading_function(self, signals_df):
        """
        The trading_function function is the main function that executes trades.
        It takes in a signals_df dataframe and iterates through each row, checking for buy/sell signals.
        If there are any buy/sell signals, it will execute the trade accordingly.

        :param self: Refer to the class itself
        :param signals_df: Pass the dataframe of signals to the trading_function function
        :return: A list of dictionaries
        :doc-author: Trelent
        """

        try:
            crypto_positions = await r.get_crypto_positions()
            for index, row in signals_df.iterrows():
                if row['buy_signal']:
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
                    buying_power = await self.update_buying_power()
                    if buying_power > 0:
                        await r.order_buy_crypto_limit(symbol=row['coin'],
                                                    quantity = buying_power / row['close'],
                                                    limitPrice = row['close'],
                                                    timeInForce = 'gtc')
                        self.logger.info(f'Bought {row["coin"]} at {row["close"]}.')
                if row['sell_signal']:
                    for position in crypto_positions:
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
                            await r.order_sell_crypto_limit(symbol=row['coin'],
                                                        quantity=position['quantity'],
                                                        limitPrice=row['close'],
                                                        timeInForce='gtc')
                            self.logger.info(f'Sold {row["coin"]} at {row["close"]}.')
        except Exception as e:
            self.logger.error(f'Unable to execute trades... {e}')

    async def get_total_crypto_dollars(self):
        """
        The get_total_crypto_dollars function is used to get the total value of all crypto positions in dollars.

        :param self: Allow the function to access variables that belong to the class
        :return: The total value of the crypto positions in dollars
        :doc-author: Trelent
        """

        try:
            crypto_positions = await r.get_crypto_positions()
            total_crypto_dollars = 0
            for position in crypto_positions:
                total_crypto_dollars += float(position['quantity']) * float(await r.crypto.get_crypto_quote(position['currency']['code'])['mark_price'])
            return total_crypto_dollars
        except Exception as e:
            self.logger.error(f'Unable to get total value of crypto... {e}')
            return 0

    async def update_buying_power(self):
        """
        The update_buying_power function updates the buying power of the user.
            It does this by first loading account profile information from Robinhood,
            then it gets the total crypto dollars available to spend, and finally adds
            those two values together to get a final value for buying power.

        :param self: Represent the instance of the class
        :return: The buying power of the account
        :doc-author: Trelent
        """

        try:
            profile_info = await r.load_account_profile()
            cash_available = float(profile_info['cash_available_for_withdrawal'])
            crypto_dollars = await self.get_total_crypto_dollars()
            buying_power = cash_available + crypto_dollars
            return buying_power
        except Exception as e:
            self.logger.error(f'Unable to update buying power... {e}')
            return 0

    async def check_stop_loss_prices(self, coins, stop_loss_prices):
        """
        The check_stop_loss_prices function is used to check the current price of a crypto currency against a stop loss price.
        If the current price is less than or equal to the stop loss, then all positions in that coin will be sold at market.
        This function should be called every minute.

        :param self: Access the class attributes and methods
        :param coins: Specify which coins to check for stop loss prices
        :param stop_loss_prices: Store the stop loss prices for each coin
        :return: The current price of the coin
        :doc-author: Trelent
        """

        try:
            for coin in coins:
                current_price = float(await r.crypto.get_crypto_quote(coin)['mark_price'])
                if current_price < stop_loss_prices[coin]:
                    crypto_positions = await r.get_crypto_positions()
                    for position in crypto_positions:
                        if position['currency']['code'] == coin:
                            await r.order_sell_crypto_limit(coin, position['quantity'], current_price)
                            self.logger.info(f'Sold {coin} at {current_price} due to stop loss.')
        except Exception as e:
            self.logger.error(f'Unable to check stop loss prices... {e}')

    async def main(self, coins, stop_loss_prices):
        """
        The main function is the heart of the bot. It runs every minute and performs all of the following tasks:
            1) Checks if it is daytime (between 9:30am and 4pm EST). If not, it will not run.
            2) Resets all variables to their default values in case they were changed by a previous iteration.
            3) Runs calculate_ta_indicators() to generate signals for each coin in coins list based on technical indicators.
               This function returns a dataframe with columns 'coin', 'signal', 'price' for each coin that has generated a signal,
               as well

        :param self: Refer to the object itself
        :param coins: Pass the list of coins that you want to trade
        :param stop_loss_prices: Check if the price of a coin is below the stop loss price
        :return: A dataframe of signals
        :doc-author: Trelent
        """

        try:
            utility = Utility()
            if utility.is_daytime():
                await self.resetter()
                signals_df = await self.calculate_ta_indicators(coins)
                await self.trading_function(signals_df)
                await self.check_stop_loss_prices(coins, stop_loss_prices)
            else:
                self.logger.info('It is not daytime. The main function will not run.')
        except Exception as e:
            self.logger.error(f'Unable to run main function... {e}')

class Looper:
    def __init__(self, trader: Trader):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the trader and logger objects, which are used throughout the class.

        :param self: Represent the instance of the class
        :param trader: Trader: Pass the trader object into the looper class
        :return: An instance of the looper class
        :doc-author: Trelent
        """

        self.trader = trader
        self.logger = logging.getLogger('looper')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    async def run_async_functions(self, loop_count, coins, stop_loss_prices):
        """
        The run_async_functions function is a recursive function that will run the main function of the trader class
            every 60 seconds. The loop_count variable is used to keep track of how many times this function has been called,
            and if it's divisible by 10, then an error message will be logged. This helps prevent spamming the log file with
            errors.

        :param self: Represent the instance of the class
        :param loop_count: Keep track of how many times the function has been called
        :param coins: Specify which coins to trade
        :param stop_loss_prices: Pass the stop loss prices to the main function in trader
        :return: A dataframe of trading signals
        :doc-author: Trelent
        """

        try:
            await self.trader.main(coins, stop_loss_prices)
            await asyncio.sleep(60)
            loop_count += 1
            await self.run_async_functions(loop_count, coins, stop_loss_prices)
        except Exception as e:
            if loop_count % 10 == 0:
                await self.logger.error(f'Unable to generate trading signals... {e}')
            return pd.DataFrame()

    async def trading_function(self, signals_df):
        """
        The trading_function function is the main function that executes trades.
        It takes in a signals_df dataframe and iterates through each row, checking for buy/sell signals.
        If there are any buy/sell signals, it will execute the trade accordingly.

        :param self: Refer to the object itself
        :param signals_df: Pass the dataframe containing all of the signals for each coin
        :return: A list of dictionaries
        :doc-author: Trelent
        """

        try:
            crypto_positions = await r.get_crypto_positions()
            for index, row in signals_df.iterrows():
                if row['buy_signal']:
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
                    buying_power = await self.update_buying_power()
                    if buying_power > 0:
                        await r.order_buy_crypto_limit(symbol=row['coin'],
                                                    quantity = buying_power / row['close'],
                                                    limitPrice = row['close'],
                                                    timeInForce = 'gtc')
                        self.logger.info(f'Bought {row["coin"]} at {row["close"]}.')
                if row['sell_signal']:
                    for position in crypto_positions:
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
                            await r.order_sell_crypto_limit(symbol=row['coin'],
                                                        quantity=position['quantity'],
                                                        limitPrice=row['close'],
                                                        timeInForce='gtc')
                            self.logger.info(f'Sold {row["coin"]} at {row["close"]}.')
        except Exception as e:
            self.logger.error(f'Unable to execute trades... {e}')

    async def get_total_crypto_dollars(self):
        """
        The get_total_crypto_dollars function is used to get the total value of all crypto positions in dollars.

        :param self: Represent the instance of the class
        :return: The total value of your crypto positions
        :doc-author: Trelent
        """

        try:
            crypto_positions = await r.get_crypto_positions()
            total_crypto_dollars = 0
            for position in crypto_positions:
                total_crypto_dollars += float(position['quantity']) * float(await r.crypto.get_crypto_quote(position['currency']['code'])['mark_price'])
            return total_crypto_dollars
        except Exception as e:
            self.logger.error(f'Unable to get total value of crypto... {e}')
            return 0

    async def update_buying_power(self):
        """
        The update_buying_power function updates the buying power of the user.
            It does this by first loading account profile information from Robinhood,
            then it gets the total crypto dollars available to spend, and finally adds
            those two values together to get a final value for buying power.

        :param self: Access the class attributes and methods
        :return: A float
        :doc-author: Trelent
        """

        try:
            profile_info = await r.load_account_profile()
            cash_available = float(profile_info['cash_available_for_withdrawal'])
            crypto_dollars = await self.get_total_crypto_dollars()
            buying_power = cash_available + crypto_dollars
            return buying_power
        except Exception as e:
            self.logger.error(f'Unable to update buying power... {e}')
            return 0

    async def check_stop_loss_prices(self, coins, stop_loss_prices):
        """
        The check_stop_loss_prices function is used to check the current price of a crypto currency against a stop loss price.
        If the current price is less than or equal to the stop loss, then all positions in that coin will be sold at market.
        This function should be called every minute.

        :param self: Represent the instance of the class
        :param coins: Specify which coins to check
        :param stop_loss_prices: Store the stop loss prices for each coin
        :return: A boolean value
        :doc-author: Trelent
        """

        try:
            for coin in coins:
                current_price = float(await r.crypto.get_crypto_quote(coin)['mark_price'])
                if current_price < stop_loss_prices[coin]:
                    crypto_positions = await r.get_crypto_positions()
                    for position in crypto_positions:
                        if position['currency']['code'] == coin:
                            await r.order_sell_crypto_limit(coin, position['quantity'], current_price)
                            self.logger.info(f'Sold {coin} at {current_price} due to stop loss.')
        except Exception as e:
            self.logger.error(f'Unable to check stop loss prices... {e}')

    async def main(self, coins, stop_loss_prices):
        """
        The main function is the heart of the bot. It runs every minute and performs all of the following tasks:
            1) Checks if it is daytime (between 9:30am and 4pm EST). If not, it will not run.
            2) Resets all variables to their default values in case they were changed by a previous iteration.
            3) Runs calculate_ta_indicators() to generate signals for each coin in coins list based on technical indicators.
               This function returns a dataframe with columns 'coin', 'signal', 'price' for each coin that has generated a signal,
               as well

        :param self: Access the class attributes and methods
        :param coins: Pass in a list of coins to the main function
        :param stop_loss_prices: Check if the price of a coin has dropped below the stop loss price
        :return: A dataframe
        :doc-author: Trelent
        """

        try:
            utility = Utility()
            if utility.is_daytime():
                await self.update_buying_power()
                signals_df = await self.calculate_ta_indicators(coins)
                crypto_positions = await r.get_crypto_positions()
                await self.execute_trades(signals_df, crypto_positions)
                await self.check_stop_loss_prices(coins, stop_loss_prices)
            else:
                self.logger.info('It is not daytime. The main function will not run.')
        except Exception as e:
            self.logger.error(f'Unable to run main function... {e}')

    async def main_looper(self, coins, stop_loss_prices):
        """
        The main_looper function is used to run the main function every minute.

        :param self: Represent the instance of the class
        :param coins: Pass in a list of coins to the main function
        :param stop_loss_prices: Check if the price of a coin has dropped below the stop loss price
        :return: None
        :doc-author: Trelent
        """

        try:
            while True:
                await self.main(coins, stop_loss_prices)
                await asyncio.sleep(60)
        except Exception as e:
            self.logger.error(f'Unable to run main looper... {e}')

import json
import asyncio
from trader import Trader
from looper import Looper

def calculate_stop_loss_prices(coins, stop_loss_percent):
    stop_loss_prices = {}
    for coin in coins:
        mark_price = float(r.crypto.get_crypto_quote(coin)['mark_price'])
        stop_loss_price = mark_price - (mark_price * stop_loss_percent)
        stop_loss_prices[coin] = stop_loss_price
    return stop_loss_prices

def load_credentials(file_path):
    with open(file_path) as f:
        return json.load(f)

def main():
    stop_loss_percent = 0.05
    coins = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']

    credentials = load_credentials('config/credentials.json')
    trader = Trader(username=credentials['username'], password=credentials['password'])

    stop_loss_prices = calculate_stop_loss_prices(coins, stop_loss_percent)
    print(f'Stop loss prices: {stop_loss_prices}')

    looper = Looper(trader)
    asyncio.run(looper.main_looper(coins, stop_loss_prices))











if __name__ == '__main__':
    stop_loss_percent = 0.05
    coins = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']

    with open('config/credentials.json') as f:
        credentials = json.load(f)
    username = credentials['username']
    password = credentials['password']
    # login to Robinhood and set up a Trader Object
    trader_object = Trader(username='username', password='password')
    #^ set stop losses for each coin by multiplying the current price by the stop loss percent (0.05) and subtracting that from the current price (to get the stop loss price).
    stop_loss_prices = {coin: float(r.crypto.get_crypto_quote(coin)['mark_price']) - (float(r.crypto.get_crypto_quote(coin)['mark_price']) * stop_loss_percent) for coin in coins}
    print(f'Stop loss prices: {stop_loss_prices}')

    trader = Trader('username', 'password')
    looper = Looper(trader)
    asyncio.run(looper.main_looper(coins, stop_loss_prices))
