import robin_stocks as r
import pandas as pd
import numpy as np
import logging
from colorama import Fore, Style
from tqdm import tqdm
import asyncio

class Utility:
    def __init__(self):
        self.logger = logging.getLogger('utility')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def is_daytime(self):
        return True

    def log_file_size_checker(self):
        pass

class Trader:
    def __init__(self):
        self.logger = logging.getLogger('trader')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def resetter(self):
        pass

    def calculate_ta_indicators(self, coins):
        return pd.DataFrame()

    def trading_function(self, signals_df):
        try:
            crypto_positions = r.get_crypto_positions()
            for index, row in signals_df.iterrows():
                if row['buy_signal']:
                    self.execute_buy(row, crypto_positions)
                if row['sell_signal']:
                    self.execute_sell(row, crypto_positions)
        except Exception as e:
            self.logger.error(f'Unable to execute trades... {e}')

    def execute_buy(self, row, crypto_positions):
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

    def execute_sell(self, row, crypto_positions):
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
        for position in crypto_positions:
            if position['currency']['code'] == row['coin']:
                r.order_sell_crypto_limit(symbol=row['coin'],
                                            quantity=position['quantity'],
                                            limitPrice=row['close'],
                                            timeInForce='gtc')
                self.logger.info(f'Sold {row["coin"]} at {row["close"]}.')

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
        except Exception as e:
            self.logger.error(f'Unable to run main function... {e}')

class Looper:
    def __init__(self, trader: Trader):
        self.trader = trader
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
            self.trader.log_file_size_checker()
        except Exception as e:
            self.logger.error(f'Unable to run async functions... {e}')

    async def main_looper(self, coins, stop_loss_prices):
        loop_count = 0
        while True:
            try:
                await self.run_async_functions(loop_count, coins, stop_loss_prices)
                loop_count += 1
                await asyncio.sleep(3600)
            except Exception as e:
                self.logger.error(f'Unable to run main looper... {e}')

if __name__ == '__main__':
    stop_loss_percent = 0.05
    coins = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']
    stop_loss_prices = {coin: float(r.crypto.get_crypto_quote(coin)['mark_price']) - (float(r.crypto.get_crypto_quote(coin)['mark_price']) * stop_loss_percent) for coin in coins}
    print(f'Stop loss prices: {stop_loss_prices}')
    trader = Trader()
    looper = Looper(trader)
    asyncio.run(looper.main_looper(coins, stop_loss_prices))
