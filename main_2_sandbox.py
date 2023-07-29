import logging
# import ThreadPoolExecutor
import concurrent.futures as futures
from concurrent.futures import ThreadPoolExecutor
# traceback
import traceback
import pandas as pd
import pandas_ta as ta
import robin_stocks as rstocks
from robin_stocks import robinhood as r
from datetime import datetime
from pytz import timezone
import asyncio
import os
from legacy.V5.main2 import stop_loss_percent
from tqdm import tqdm
from colorama import Fore, Back, Style
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import numpy as np
sktrading = False
loop_count = 0 # initialized to 0
class Utility:
    """
    The Utility class provides functions for logging into Robinhood, resetting orders, generating trading signals, executing actions based on these signals, updating the buying power, and checking stop loss prices.
    """
    def __init__(self):
        self.logger = logging.getLogger('utility')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    async def log_file_size_checker(self):
        """
        The log_file_size_checker function is an async function that checks the size of the log file and removes lines from the start of the file to maintain a rolling log of 1000 lines.
        :return: None
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
        self.logger = logging.getLogger('trader')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
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
                macd_data = ta.macd(df.close)  # 0 = macd_line, 1 = signal_line, 2 = macd_hist
                df['macd_line'] = macd_data[0]
                df['signal_line'] = macd_data[1]
                df['macd_hist'] = macd_data[2]
                df['rsi'] = ta.rsi(df.close)
                df['macd_line'], df['signal_line'], df['macd_hist'] = ta.macd(df.close)
                df['rsi'] = ta.rsi(df.close)
                df['buy_signal'] = ((df.macd_line > df.signal_line) & (df.rsi < 30)) | ((df.stochastic_k > df.stochastic_d) & (df.williams < -80))
                df['sell_signal'] = ((df.macd_line < df.signal_line) & (df.rsi > 70)) | ((df.stochastic_k < df.stochastic_d) & (df.williams > -20))
                signals_df = signals_df.append(df)
            # Prepare the data
            with ThreadPoolExecutor() as executor:
                executor.map(self.get_data_and_train, coins)
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
                    buying_power = self.update_buying_power()
                    if buying_power > 0:
                        r.order_buy_crypto_limit(symbol=row['coin'],
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
                            r.order_sell_crypto_limit(coin, position['quantity'], current_price)
                            self.logger.info(f'Sold {coin} at {current_price} due to stop loss.')
        except Exception as e:
            self.logger.error(f'Unable to check stop loss prices... {e}')
    def sklearner(self, signals_df, coins, stop_loss_prices, utility):
        print('sklearner')

        # if the folder for model files does not exist, create it
        if not os.path.exists('models'):
            os.makedirs('models')

        # Prepare the data
        for coin in tqdm(coins):
            df = utility.get_last_100_days(coin)

            # Prepare the features and target variable
            features, target = self.prepare_features_and_target(df)

            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            # Train the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # generate the predicted signals for the coin
            df['predicted_close'] = np.nan
            df.iloc[len(df) - len(predictions):, df.columns.get_loc('predicted_close')] = predictions

            # calculate the mean squared error
            mse = mean_squared_error(y_test, predictions)
            self.logger.info(f'MSE: {mse}')

            # if the mse is less than the industry std of 0.5, save the model
            if mse < 0.5:
                self.logger.info(f'Saving model for {coin}...')
                print(f'Recommended Action for {coin}: is to, {model.predict(df.tail(1))[0]}')
            # Save the model to a file
            with open(f'models/{coin}.pkl', 'wb') as f:
                pickle.dump(model, f)
    def prepare_features_and_target(self, df):
        """
        Prepares the features and target variable for training the model.
        """
        # Add a column that contains the target variable
        df['close2'] = df['close'].shift(-1)

        # Drop the last row since it will be NaN
        df = df[:-1]

        # Use the 'close' column as the feature and drop the 'date' column since it is not a feature
        features = df.drop('close', axis=1)
        features = features.drop('date', axis=1)

        # The target variable is the 'close2' column
        target = df['close2']

        return features, target
    def get_data_and_train(self, coin):
        df = self.utility.get_last_100_days(coin)
        print(f"Getting the historical data for {coin}...")

        # Prepare the features and target variable
        features, target = self.prepare_features_and_target(df)

        # Train the model
        self.sklearner(features, target, coin)
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
        self.logger = logging.getLogger('looper')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    async def run_sklearner(self, signals_df, coins, stop_loss_prices, loop_count):
        """
        The run_sklearner function runs the sklearner function asynchronously.
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :param loop_count: Keep track of how many times the loop has run
        :return: A coroutine object
        :doc-author: Trelent
        """
        try:
            await asyncio.sleep(1)
            print('Running sklearner...')
            self.trader.sklearner(signals_df, coins, stop_loss_prices, utility)
        except Exception as e:
            self.logger.error(f'Unable to run sklearner... {e}')
            print(traceback.format_exc())
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
            self.trader.log_file_size_checker()
            self.run_sklearner(self.trader.signals_df, coins, stop_loss_prices, loop_count)
            loop_count += 1
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
                await asyncio.sleep(3600)
            except Exception as e:
                self.logger.error(f'Error in main loop... {e}')
if __name__ == '__main__':
    stop_loss_percent = 0.05
    coins = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'AVAX', 'XLM', 'BCH', 'XTZ']
    stop_loss_prices = {coin: float(r.crypto.get_crypto_quote(coin)['mark_price']) - (float(r.crypto.get_crypto_quote(coin)['mark_price']) * stop_loss_percent) for coin in coins}
    print(f'Stop loss prices: {stop_loss_prices}')
    trader = Trader()
    looper = Looper(trader)
    asyncio.run(looper.main_looper(coins, stop_loss_prices))