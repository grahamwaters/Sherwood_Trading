import logging
import pandas as pd
import robin_stocks as rstocks
from robin_stocks import robinhood as r
from datetime import datetime
import configparser
from pytz import timezone
import asyncio
from tqdm import tqdm
from colorama import Fore, Style
import pandas_ta as ta

# Constants
STOP_LOSS_PERCENT = 0.05
COINS = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']

# Utility class
class Utility:
    def __init__(self):
        pass

    async def log_file_size_checker(self):
        while True:
            with open('logs/robinhood.log', 'r') as f:
                lines = f.readlines()
                if len(lines) > 1000:  # if the log file is greater than 1000 lines
                    num_lines_to_remove = len(lines) - 1000
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
        return 8 <= current_hour <= 20

    def get_stop_loss_price(self, coin):
        last_price = float(r.crypto.get_crypto_quote(coin)['mark_price'])
        return last_price - (last_price * STOP_LOSS_PERCENT)

# Trader class
class Trader:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.utility = Utility()
        self.setup_logger()
        self.login_setup()
        self.stop_loss_prices = self.calculate_stop_loss_prices()

    def login_setup(self):
        try:
            r.login(self.username, self.password)
            self.logger.info('Logged in to Robinhood successfully.')
        except Exception as e:
            self.logger.error(f'Unable to login to Robinhood... {e}')

    def calculate_stop_loss_prices(self):
        try:
            return {coin: float(r.crypto.get_crypto_quote(coin)['mark_price']) - (float(r.crypto.get_crypto_quote(coin)['mark_price']) * STOP_LOSS_PERCENT) for coin in COINS}
        except Exception as e:
            self.logger.error(f'Unable to calculate stop loss prices... {e}')
            return {}

    def setup_logger(self):
        self.logger = logging.getLogger('trader')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

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
        signals_df = pd.DataFrame()
        for coin in coins:
            try:
                data = self.get_historical_data(coin)
                data = self.calculate_technical_indicators(data)
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
        data.ta.ema(length=12, append=True)  # Exponential Moving Average
        data.ta.ema(length=26, append=True)  # Exponential Moving Average
        data.ta.macd(fast=12, slow=26, append=True)  # Moving Average Convergence Divergence
        data.ta.rsi(length=14, append=True)  # Relative Strength Index
        data.ta.willr(length=14, append=True)  # Williams %R
        data.ta.stoch(length=14, append=True)  # Stochastic Oscillator
        data.ta.bbands(length=20, append=True)  # Bollinger Bands
        data.ta.sma(length=20, append=True)  # Simple Moving Average
        return data

    def generate_trading_signals(self, signals, coin):
        # fill signals with data from the last row of the dataframe
        signals['coin'] = coin
        signals['close'] = data['close'].iloc[-1]
        signals['ema_12'] = data['EMA_12_ema'].iloc[-1]
        signals['ema_26'] = data['EMA_26_ema'].iloc[-1]
        signals['macd_line'] = data['MACD_12_26'].iloc[-1]
        signals['signal_line'] = data['MACDs_12_26_9'].iloc[-1]
        signals['rsi'] = data['RSI_14'].iloc[-1]
        signals['willr'] = data['WILLR_14'].iloc[-1]
        signals['stoch'] = data['STOCHk_14_3_3'].iloc[-1]
        signals['bbands_lower'] = data['BBL_20_2.0'].iloc[-1]
        signals['bbands_middle'] = data['BBM_20_2.0'].iloc[-1]
        signals['bbands_upper'] = data['BBU_20_2.0'].iloc[-1]
        signals['sma'] = data['SMA_20'].iloc[-1]
        signals['buy_signal'] = self.generate_buy_signal(signals)
        signals['sell_signal'] = self.generate_sell_signal(signals)
        return signals

    def generate_buy_signal(self, signals):
        return (
            signals['ema_12'] > signals['ema_26'] and
            signals['macd_line'] > signals['signal_line'] and
            signals['rsi'] > 50 and
            signals['willr'] > -50 and
            signals['stoch'] > 50 and
            signals['close'] < signals['bbands_lower'] and
            signals['close'] < signals['sma']
        )

    def generate_sell_signal(self, signals):
        return signals['close'] > signals['bbands_middle'] or signals['close'] > signals['bbands_upper']

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
            if self.utility.is_daytime():
                self.resetter()
                signals_df = self.calculate_ta_indicators(coins)
                self.trading_function(signals_df)
                self.check_stop_loss_prices(coins, stop_loss_prices)
            else:
                # self.logger.info('It is not daytime. The main function will not run.')
                # during nighttime we want to decrease our stakes, and the frequency of our trades
                # note: gpt, make this change in the future
                self.resetter()
                signals_df = self.calculate_ta_indicators(coins)
                self.trading_function(signals_df)
                self.check_stop_loss_prices(coins, stop_loss_prices)
        except Exception as e:
            self.logger.error(f'Unable to run main function... {e}')

# Looper class
class Looper:
    def __init__(self, trader: Trader):
        self.trader = trader
        self.setup_logger()

    def setup_logger(self):
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
            await self.trader.utility.log_file_size_checker()
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

# Main function
def main():
    config = configparser.ConfigParser()
    config.read('config/credentials.ini')
    username = config['credentials']['username']
    password = config['credentials']['password']

    trader = Trader(username, password)

    stop_loss_prices = {coin: float(r.crypto.get_crypto_quote(coin)['mark_price']) - (float(r.crypto.get_crypto_quote(coin)['mark_price']) * STOP_LOSS_PERCENT) for coin in COINS}

    looper = Looper(trader)
    asyncio.run(looper.main_looper(COINS, stop_loss_prices))

# Run the program
if __name__ == '__main__':
    COINS = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']
    STOP_LOSS_PERCENT = 0.05
    main()
