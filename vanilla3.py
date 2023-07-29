import logging
import pandas as pd
import pandas_ta as ta
import robin_stocks as rstocks
from robin_stocks import robinhood as r
from datetime import datetime
import configparser
from pytz import timezone
import asyncio
# from legacy.V5.main2 import stop_loss_percent
stop_loss_percent = 0.05
from tqdm import tqdm
from colorama import Fore, Back, Style
from legacy.V5.main import trading_function
class Utility:
    def __init__(self):
        pass
    async def log_file_size_checker():
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
        current_time = datetime.now(timezone('US/Central'))
        current_hour = current_time.hour
        if current_hour >= 8 and current_hour <= 20:
            return True
        else:
            return False
class Trader:

    def __init__(self, username, password):
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
        try:
            r.login(self.username, self.password)
            self.logger.info('Logged in to Robinhood successfully.')
        except Exception as e:
            self.logger.error(f'Unable to login to Robinhood... {e}')
    def resetter(self):
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
        utility = Utility()
        signals_df = pd.DataFrame()

        for coin in coins:
            try:
                data = self.get_historical_data(coin)
                self.calculate_technical_indicators(data)
                signals = self.generate_trading_signals(data, coin)
                signals_df = signals_df.append(signals)
            except Exception as e:
                self.logger.error(f'Unable to generate trading signals for {coin}... {e}')

        return signals_df

    def get_historical_data(self, coin):
        data = pd.DataFrame(r.crypto.get_crypto_historicals(coin, interval='5minute', span='day'))
        data = data.rename(columns={'begins_at': 'date'})
        data = data.set_index('date')
        data = data.drop(columns=['session', 'interpolated', 'symbol'])
        data = data.apply(pd.to_numeric)
        data = data.astype(float)
        data = data.sort_index(ascending=True)
        return data

    def calculate_technical_indicators(self, data):
        data.ta.ema(length=12, append=True) # Exponential Moving Average
        data.ta.ema(length=26, append=True) # Exponential Moving Average
        data.ta.macd(fast=12, slow=26, append=True) # Moving Average Convergence Divergence
        data.ta.rsi(length=14, append=True) # Relative Strength Index
        data.ta.willr(length=14, append=True) # Williams %R
        data.ta.stoch(length=14, append=True) # Stochastic Oscillator
        data.ta.bbands(length=20, append=True) # Bollinger Bands
        data.ta.sma(length=20, append=True) # Simple Moving Average
        #data.ta.sar(append=True) # Parabolic SAR

    def log_file_size_checker(self):
        while True:
            with open('logs/robinhood.log') as f:
                lines = f.readlines()
                if len(lines) > 1000:
                    # find how many lines to remove
                    num_lines_to_remove = len(lines) - 1000
                    # remove the first num_lines_to_remove lines
                    with open('logs/robinhood.log', 'w') as f:
                        f.writelines(lines[num_lines_to_remove:])

    def generate_trading_signals(self, coin):
        # Create a DataFrame with all of the data and indicators
        signals = pd.DataFrame(index=self.data.index)
        signals['close'] = self.data['close_price']
        signals['open'] = self.data['open_price']
        signals['high'] = self.data['high_price']
        signals['low'] = self.data['low_price']
        signals['ema_12'] = self.data.ta.ema(length=12)
        signals['ema_26'] = self.data.ta.ema(length=26)
        signals['macd_line'] = self.data.ta.macd(fast=12, slow=26)
        signals['signal_line'] = self.data.ta.macd(fast=12, slow=26, signal=9)
        signals['rsi'] = self.data.ta.rsi(length=14)
        return signals

    def generate_buy_signal(self, row):
        return (
            row['close'] < row['bbands_middle'] or
            row['close'] < row['bbands_lower'] ) or (
            row['rsi'] < 30 or \
            row['willr'] < -80 or \
            row['stoch'] < 20 or \
            row['close'] < row['sma'] or \
            row['macd_line'] > row['signal_line'] or \
            row['ema_12'] > row['ema_26']
        )

    def generate_sell_signal(self, row):
        return row['close'] > row['bbands_middle'] or row['close'] > row['bbands_upper']


    def generate_buy_signal(self, signals):
        return (

            signals['close'] < signals['bbands_middle'] or
            signals['close'] < signals['bbands_lower'] ) or (
            signals['rsi'] < 30 or \
            signals['willr'] < -80 or \
            signals['stoch'] < 20 or \
            signals['close'] < signals['sma'] or \
            signals['macd_line'] > signals['signal_line'] or \
            signals['ema_12'] > signals['ema_26']
            )

    def generate_sell_signal(self, signals):
        return (
            signals['close'] > signals['bbands_middle'] or
            signals['close'] > signals['bbands_upper'] ) or (
            signals['rsi'] > 70 or \
            signals['willr'] > -20 or \
            signals['stoch'] > 80 or \
            signals['close'] > signals['sma'] or \
            signals['macd_line'] < signals['signal_line'] or \
            signals['ema_12'] < signals['ema_26']
            )

    def trading_function(self, signals_df):
        try:
            crypto_positions = r.get_crypto_positions()
            for position in crypto_positions:
                if position['currency']['code'] in signals_df['coin'].values:
                    if signals_df[signals_df['coin'] == position['currency']['code']]['sell_signal'].values[0]:
                        self.logger.info(f'Selling {position["currency"]["code"]}...')
                        r.order_sell_crypto_limit(position['currency']['code'], position['quantity'], position['cost_bases'][0]['direct_cost_basis'])
                        self.logger.info(f'Sold {position["currency"]["code"]}...')
                    else:
                        self.logger.info(f'Not selling {position["currency"]["code"]}...')
            for index, row in signals_df.iterrows():
                if row['buy_signal']:
                    self.logger.info(f'Buying {row["coin"]}...')
                    r.order_buy_crypto_by_price(row['coin'], self.get_buying_power() / len(signals_df), 'price')
                    self.logger.info(f'Bought {row["coin"]}...')
        except Exception as e:
            self.logger.error(f'Unable to execute trades... {e}')


    def get_total_crypto_dollars(self):
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
    def main(self, coins, stop_loss_prices):
        try:
            utility = Utility()
            if utility.is_daytime():
                self.resetter()
                signals_df = self.calculate_ta_indicators(coins)
                self.trading_function(signals_df)
                self.check_stop_loss_prices(coins, stop_loss_prices)
            else:
                self.logger.info('It is not daytime. The main function will not run.')
                self.resetter()
                signals_df = self.calculate_ta_indicators(coins)
                self.trading_function(signals_df)
                self.check_stop_loss_prices(coins, stop_loss_prices)
        except Exception as e:
            self.logger.error(f'Unable to run main function... {e}')
class Looper:
    def __init__(self, trader: Trader):
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
                self.trader.update_buying_power()
            self.trader.main(coins, stop_loss_prices)
            # run all async functions simultaneously
            # log_file_size_checker included to prevent log file from getting too large
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
            except Exception as e:
                self.logger.error(f'Error in main loop... {e}')
# run the program
if __name__ == '__main__':
    stop_loss_percent = 0.05 #^ set the stop loss percent at 5% (of the invested amount)
    coins = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ'] #^ set the coins to check
    #^ set stop losses for each coin by multiplying the current price by the stop loss percent (0.05) and subtracting that from the current price (to get the stop loss price).

    # config/credentials.ini
    config = configparser.ConfigParser()
    config.read('config/credentials.ini')
    # Robinhood login
    r.login(username=config['credentials']['username'], password=config['credentials']['password'])
    #^ get the current price of each coin
    stop_loss_prices = {coin: float(r.crypto.get_crypto_quote(coin)['mark_price']) - (float(r.crypto.get_crypto_quote(coin)['mark_price']) * stop_loss_percent) for coin in coins}
    print(f'Stop loss prices: {stop_loss_prices}')
    # set up logging
    log_file_name = 'logs/main.log'
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file_name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    trader = Trader(
        username=config['credentials']['username'],
        password=config['credentials']['password']
        ) #^ create an instance of the Trader class

    looper = Looper(trader) #^ create an instance of the Looper class (which will run the Trader class)
    asyncio.run(looper.main_looper(coins, stop_loss_prices)) #^ run the main_looper function