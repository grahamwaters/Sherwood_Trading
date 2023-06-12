import asyncio
import logging
import os
import datetime
import pytz
import json
import pandas as pd
from colorama import Fore, Style
from robin_stocks import robinhood as r
logging.basicConfig(level=logging.INFO)
def load_files():
    with open('strategy.json', 'r') as f:
        strategy = json.load(f)
    with open('secrets.json', 'r') as f:
        credentials = json.load(f)
    return strategy, credentials
def login_to_robinhood(credentials):
    r.login(credentials['username'], credentials['password'])
class CryptoBot:
    def __init__(self, strategy):
        self.strategy = strategy
        self.portfolio = pd.DataFrame()
        self.coin_positions = pd.DataFrame()
    async def get_account_info(self):
        accountinfo = r.build_user_profile()
        accountinfo['timestamp'] = datetime.datetime.now(pytz.timezone('US/Eastern'))
        accountinfo['cash'] = float(accountinfo['cash'])
        accountinfo['crypto_buying_power'] = float(accountinfo['cash'])
        return accountinfo
    async def get_coin_positions(self):
        coin_positions = r.get_crypto_positions()
        coin_positions = pd.DataFrame(coin_positions)
        coin_positions['quantity'] = float(coin_positions['quantity'][0])
        return coin_positions
    async def fill_portfolio(self):
        self.portfolio = await self.get_account_info()
        self.coin_positions = await self.get_coin_positions()
    async def fetch_price_from_api(self, ticker: str) -> float:
        price_history = r.get_crypto_historicals(ticker, interval='5minute', span='day')
        return float(price_history[-1]['close_price'])
    async def get_price_history(self, ticker: str) -> list:
        price_history = r.get_crypto_historicals(ticker, interval='5minute', span='day')
        return [float(bar['close_price']) for bar in price_history]
    async def sleep(self, duration):
        await asyncio.sleep(duration)
class BankerBot(CryptoBot):
    def __init__(self, strategy):
        super().__init__(strategy)
        self.coin_positions = pd.DataFrame()
        self.portfolio = pd.DataFrame()
    async def get_coin_orders(self):
        coin_orders = r.get_all_crypto_orders()
        coin_orders = pd.DataFrame(coin_orders)
        return coin_orders
class AnalystBot(CryptoBot):
    def __init__(self, config):
        super().__init__(config)
        self.currencies = config['currencies']
        self.bot_config = config['botConfiguration']
    async def calculate_moving_average(self, ticker):
        price_history = await self.get_price_history(ticker)
        price_history = price_history[:-1]
        moving_average = sum(price_history) / len(price_history)
        return moving_average
    async def calculate_rsi(self, ticker):
        price_history = await self.get_price_history(ticker)
        price_change = [price_history[i] - price_history[i-1] for i in range(1, len(price_history))]
        average_gain = sum([gain for gain in price_change if gain > 0]) / len(price_change)
        average_loss = sum([loss for loss in price_change if loss < 0]) / len(price_change)
        relative_strength = average_gain / average_loss
        relative_strength_index = 100 - (100 / (1 + relative_strength))
        return relative_strength_index
    async def calculate_bollinger_bands(self, ticker):
        price_history = await self.get_price_history(ticker)
        moving_average = await self.calculate_moving_average(ticker)
        standard_deviation = sum([(price - moving_average) ** 2 for price in price_history]) / len(price_history)
        upper_band = moving_average + (2 * standard_deviation)
        lower_band = moving_average - (2 * standard_deviation)
        return upper_band, lower_band
    async def calculate_macd(self, ticker):
        price_history = await self.get_price_history(ticker)
        exponential_moving_average_12 = sum(price_history[:12]) / 12
        exponential_moving_average_26 = sum(price_history[:26]) / 26
        moving_average_convergence_divergence = exponential_moving_average_12 - exponential_moving_average_26
        return moving_average_convergence_divergence
    async def analyze_coin(self, ticker):
        moving_average = await self.calculate_moving_average(ticker) if self.bot_config['technicalIndicators']['movingAverage'] else None
        rsi = await self.calculate_rsi(ticker) if self.bot_config['technicalIndicators']['relativeStrengthIndex'] else None
        bollinger_bands = await self.calculate_bollinger_bands(ticker) if self.bot_config['technicalIndicators']['bollingerBands'] else None
        macd = await self.calculate_macd(ticker) if self.bot_config['technicalIndicators']['macd'] else None
        if moving_average and rsi and bollinger_bands and macd:
            if moving_average < bollinger_bands[1] and rsi < 30 and macd < 0:
                return 'buy'
            elif moving_average > bollinger_bands[0] and rsi > 70 and macd > 0:
                return 'sell'
            else:
                return 'hold'
        else:
            return 'hold'
    def load_strategy(self):
        with open('strategy.json') as f:
            self.strategy = json.load(f)
        return self.strategy
class MerchantBot(CryptoBot):
    def __init__(self, strategy):
        super().__init__(strategy)
        self.start_time = datetime.datetime.now()
        self.analyst_bot = AnalystBot(self.strategy)
    def get_quantity(self, ticker):
        positions = r.get_crypto_positions()
        for position in positions:
            if position['currency']['code'] == ticker:
                return float(position['quantity'])
        return 0
    async def trade_coin(self, ticker):
        signal = await self.analyst_bot.analyze_coin(ticker)
        quantity = self.get_quantity(ticker)
        print(Fore.YELLOW + f"[{datetime.datetime.now()}] {ticker} quantity: {quantity}" + Fore.RESET)
        signal = await AnalystBot.analyze_coin(ticker=ticker)
        elapsed_time = datetime.datetime.now() - self.start_time
        if elapsed_time.total_seconds() < 3600:
            if quantity > 0 and signal == 'sell':
                print(f"[PAPER TRADE] Would sell {quantity} of {ticker}")
            elif quantity > 0 and signal == 'buy':
                print(f"[PAPER TRADE] Would sell {quantity * 0.60} of {ticker}")
            elif quantity == 0 and signal == 'buy':
                print(f"[PAPER TRADE] Would buy {ticker}")
        else:
            if quantity > 0 and signal == 'sell':
                r.order_sell_crypto_by_quantity(ticker, quantity, timeInForce='gtc')
            elif quantity > 0 and signal == 'buy':
                r.order_buy_crypto_by_price(ticker, 0.50, timeInForce='gtc')
            elif quantity == 0 and signal == 'buy':
                r.order_buy_crypto_by_price(ticker, 1.00, timeInForce='gtc')
def main():
    strategy, credentials = load_files()
    login_to_robinhood(credentials)
    banker = BankerBot(strategy)
    analyst = AnalystBot(strategy)
    merchant = MerchantBot(strategy)
    tasks = []
    for currency in strategy["currencies"]:
        tasks.append(analyst.analyze_coin(currency))
        tasks.append(merchant.trade_coin(currency))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))
    return r.get_crypto_positions()
if __name__ == '__main__':
    main()