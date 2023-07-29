import asyncio
import configparser
import logging
from datetime import datetime

import pandas as pd
import robin_stocks as rstocks
from colorama import Back, Fore, Style
from pytz import timezone
from robin_stocks import robinhood as r
from tqdm import tqdm

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class Trader:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.logger = setup_logger('trader')
        self.login_setup()

    def login_setup(self):
        r.login(self.username, self.password)
        self.logger.info('Logged in to Robinhood')

    def get_historical_prices(self, coin):
        return r.crypto.get_crypto_historicals(coin, interval='5minute', span='week', bounds='24_7')

    def update_buying_power(self):
        self.buying_power = r.profiles.load_account_profile('buying_power')

    def get_crypto_quote(self, coin):
        return r.crypto.get_crypto_quote(coin)

    def get_crypto_positions(self):
        return r.crypto.get_crypto_positions()

class Looper:
    def __init__(self, trader):
        self.trader = trader
        self.logger = setup_logger('looper')

    async def main_looper(self, coins, stop_loss_prices):
        while True:
            self.logger.info('Starting loop')
            await self.loop(coins, stop_loss_prices)
            self.logger.info('Finished loop')
            await asyncio.sleep(60)

    async def loop(self, coins, stop_loss_prices):
        self.trader.update_buying_power()
        self.logger.info(f'Buying power: {self.trader.buying_power}')
        self.logger.info(f'Coins: {coins}')
        self.logger.info(f'Stop loss prices: {stop_loss_prices}')
        for coin in coins:
            self.logger.info(f'Checking {coin}')
            quote = self.trader.get_crypto_quote(coin)
            self.logger.info(f'Quote: {quote}')
            if float(quote['mark_price']) < stop_loss_prices[coin]:
                self.logger.info(f'Stop loss triggered for {coin}')
                self.sell_coin(coin)
            else:
                self.logger.info(f'No stop loss triggered for {coin}')
        self.logger.info('Finished checking coins')

    def sell_coin(self, coin):
        self.logger.info(f'Selling {coin}')
        self.trader.sell_crypto_by_quantity(coin, self.trader.get_crypto_positions()[coin]['quantity'])
        self.logger.info(f'Sold {coin}')

    def buy_coin(self, coin):
        self.logger.info(f'Buying {coin}')
        self.trader.buy_crypto_by_quantity(coin, self.trader.buying_power)
        self.logger.info(f'Bought {coin}')


def get_credentials():
    config = configparser.ConfigParser()
    config.read('config/credentials.ini')
    credentials = {
        "username": config['credentials']['username'],
        "password": config['credentials']['password']
    }
    return credentials

def set_stop_loss_prices(coins, stop_loss_percent):
    stop_loss_prices = {
        coin: float(r.crypto.get_crypto_quote(coin)['mark_price']) - (float(r.crypto.get_crypto_quote(coin)['mark_price']) * stop_loss_percent)
        for coin in coins
    }
    return stop_loss_prices

if __name__ == '__main__':
    print('Starting program...')
    stop_loss_percent = 0.05
    coins = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'AVAX', 'XLM', 'BCH', 'XTZ']
    print(f'Coins: {coins}')

    credentials = get_credentials()
    trader = Trader(credentials["username"], credentials["password"])
    looper = Looper(trader)

    stop_loss_prices = set_stop_loss_prices(coins, stop_loss_percent)
    print(f'Stop loss prices: {stop_loss_prices}')
    asyncio.run(looper.main_looper(coins, stop_loss_prices))
