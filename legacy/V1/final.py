import asyncio
import json
import pandas as pd
from datetime import datetime
from pytz import timezone
from robin_stocks import robinhood as r
from colorama import Fore, Style
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry

ONMODE = True

class CryptoBot:
    def __init__(self, strategy):
        print(Fore.WHITE + "Initializing CryptoBot..." + Style.RESET_ALL)
        self.strategy = strategy
        self.portfolio = pd.DataFrame()
        self.coin_positions = pd.DataFrame()

    @sleep_and_retry
    async def get_account_info(self):
        print(Fore.WHITE + "Getting account info..." + Style.RESET_ALL)
        accountinfo = r.build_user_profile()
        accountinfo['timestamp'] = datetime.now(timezone('US/Eastern'))
        accountinfo['cash'] = float(accountinfo['cash'])
        accountinfo['crypto_buying_power'] = float(accountinfo['cash'])
        return accountinfo

    @sleep_and_retry
    async def get_coin_positions(self):
        print(Fore.WHITE + "Getting coin positions..." + Style.RESET_ALL)
        coin_positions = pd.DataFrame(r.get_crypto_positions())
        coin_positions['quantity'] = float(coin_positions['quantity'][0])
        return coin_positions

    @sleep_and_retry
    async def fill_portfolio(self):
        print(Fore.WHITE + "Filling portfolio..." + Style.RESET_ALL)
        self.portfolio = await self.get_account_info()
        self.coin_positions = await self.get_coin_positions()

    async def get_price_history(self, ticker: str, interval='5minute', span='day') -> list:
        print(Fore.WHITE + f"Getting price history for {ticker}..." + Style.RESET_ALL)
        price_history = r.get_crypto_historicals(ticker, interval=interval, span=span)
        return [float(bar['close_price']) for bar in price_history]

    async def sleep(self, duration):
        print(Fore.WHITE + f"Sleeping for {duration} seconds..." + Style.RESET_ALL)
        await asyncio.sleep(duration)


class AnalystBot(CryptoBot):
    def __init__(self, config):
        print(Fore.YELLOW + "Initializing AnalystBot..." + Style.RESET_ALL)
        super().__init__(config)
        self.currencies = config['currencies']
        self.bot_config = config['botConfiguration']
        self.current_price = None
        self.current_holdings = None
        self.last_check = None
        self.buy_signal_strength = 0
        self.sell_signal_strength = 0
        self.stop_loss = None
        self.trailing_stop_loss = None
        self.purchase_price = None

    async def get_current_holdings(self):
        print(Fore.YELLOW + "Updating current holdings..." + Style.RESET_ALL)
        self.current_holdings = await self.get_account_info()
        return self.current_holdings

    async def calculate_moving_average(self, ticker):
        #print(Fore.YELLOW + f"Calculating moving average for {ticker}..." + Style.RESET_ALL)
        price_history = await self.get_price_history(ticker)
        self.current_price = price_history[-1] # Set the current price
        self.last_check = datetime.now(timezone('US/Eastern')) # Set the last check time
        return sum(price_history[:-1]) / len(price_history[:-1])

    async def calculate_rsi(self, ticker):
        #print(Fore.YELLOW + f"Calculating RSI for {ticker}..." + Style.RESET_ALL)
        price_history = await self.get_price_history(ticker)
        price_change = [price_history[i] - price_history[i-1] for i in range(1, len(price_history))]
        average_gain = sum([gain for gain in price_change if gain > 0]) / len(price_change)
        average_loss = sum([loss for loss in price_change if loss < 0]) / len(price_change)
        relative_strength = average_gain / average_loss
        return 100 - (100 / (1 + relative_strength))

    async def calculate_bollinger_bands(self, ticker):
        #print(Fore.YELLOW + f"Calculating Bollinger Bands for {ticker}..." + Style.RESET_ALL)
        price_history = await self.get_price_history(ticker)
        moving_average = await self.calculate_moving_average(ticker)
        standard_deviation = sum([(price - moving_average) ** 2 for price in price_history]) / len(price_history) ** 0.5

        return moving_average + (2 * standard_deviation), moving_average - (2 * standard_deviation)

    async def calculate_macd(self, ticker):
        #print(Fore.YELLOW + f"Calculating MACD for {ticker}..." + Style.RESET_ALL)
        price_history = await self.get_price_history(ticker)
        exponential_moving_average_12 = sum(price_history[:12]) / 12
        exponential_moving_average_26 = sum(price_history[:26]) / 26
        return exponential_moving_average_12 - exponential_moving_average_26

    async def get_highest_price(self, ticker):
        #print(Fore.YELLOW + f"Getting highest price for {ticker}..." + Style.RESET_ALL)
        price_history = await self.get_price_history(ticker)
        return max(price_history)


    async def get_current_price(self, ticker):
        # if self.current_price is None or (datetime.now(timezone('US/Eastern')) - self.last_check).seconds > 300:
        # then get the current price
        # else return the current price (self.current_price)
        #print(Fore.YELLOW + f"Getting current price for {ticker}..." + Style.RESET_ALL)
        price_history = await self.get_price_history(ticker)
        return price_history[-1]

    async def update_highest_price(self, ticker):
        #print(Fore.YELLOW + f"Updating highest price for {ticker}..." + Style.RESET_ALL)
        price_history = await self.get_price_history(ticker)
        return max(price_history)

    async def update_current_price(self, ticker):
        #print(Fore.YELLOW + f"Updating current price for {ticker}..." + Style.RESET_ALL)
        price_history = await self.get_price_history(ticker)
        return price_history[-1]

    async def get_quantity(self, ticker):
        #print(Fore.YELLOW + f"Getting quantity for {ticker}..." + Style.RESET_ALL)
        coin_positions = await self.get_coin_positions()
        return coin_positions[coin_positions['currency'] == ticker]['quantity'].values[0]

    @sleep_and_retry
    async def analyze_coin(self, ticker):
        print(Fore.YELLOW + f"Analyzing {ticker}..." + Style.RESET_ALL)
        moving_average = await self.calculate_moving_average(ticker) if self.bot_config['technicalIndicators']['movingAverage'] else None
        rsi = await self.calculate_rsi(ticker) if self.bot_config['technicalIndicators']['relativeStrengthIndex'] else None
        bollinger_bands = await self.calculate_bollinger_bands(ticker) if self.bot_config['technicalIndicators']['bollingerBands'] else None
        macd = await self.calculate_macd(ticker) if self.bot_config['technicalIndicators']['macd'] else None
        current_price = await self.get_current_price(ticker)

        if moving_average is None or rsi is None or bollinger_bands is None or macd is None:
            print(Fore.RED + f"Unable to calculate all indicators for {ticker}." + Style.RESET_ALL)
            return 'hold'

        if self.bot_config['technicalIndicators']['trailingStopLoss']:
            highest_price = await self.get_highest_price(ticker)
            if highest_price is None:
                await self.update_highest_price(ticker, current_price)
            elif current_price < highest_price * 0.99:  # current price is less than 99% of highest price (1% drop)
                return 'sell'
            elif current_price > highest_price:
                await self.update_highest_price(ticker, current_price)
        # check that we have enough money to buy
        # if we do, then check our coins and buy 1.00 USD worth of any coins we have which have a value less than 1.00 USD
        # if we don't, then sell 1.00 USD worth of any coins we have which have a value greater than 1.00 USD
        # if we don't have any coins, then do nothing
        current_holds = await self.get_current_holdings()
        current_holdings = self.current_holdings
        current_price = await self.get_current_price(ticker)
        quantity = current_holdings[ticker]['quantity']
        if current_holds > 1.00:
            if quantity > 0:
                return 'hold'
            else:
                return 'sell'
        if moving_average < bollinger_bands[1] and rsi < 30 and macd < 0:
            print(Fore.GREEN + f"Buy signal detected for {ticker}." + Style.RESET_ALL)
            print(Fore.GREEN + f"Moving average: {moving_average}, Bollinger bands: {bollinger_bands}, RSI: {rsi}, MACD: {macd}" + Style.RESET_ALL)
            return 'buy'
        elif moving_average > bollinger_bands[0] and rsi > 70 and macd > 0:
            print(Fore.RED + f"Sell signal detected for {ticker}." + Style.RESET_ALL)
            print(Fore.RED + f"Moving average: {moving_average}, Bollinger bands: {bollinger_bands}, RSI: {rsi}, MACD: {macd}" + Style.RESET_ALL)
            return 'sell'
        elif current_price < moving_average and rsi < 30 and macd < 0:
            print(Fore.GREEN + f"Buy signal detected for {ticker}." + Style.RESET_ALL)
            print(Fore.GREEN + f"Moving average: {moving_average}, Bollinger bands: {bollinger_bands}, RSI: {rsi}, MACD: {macd}" + Style.RESET_ALL)
            return 'buy'
        elif current_price > moving_average and rsi > 70 and macd > 0:
            print(Fore.RED + f"Sell signal detected for {ticker}." + Style.RESET_ALL)
            print(Fore.RED + f"Moving average: {moving_average}, Bollinger bands: {bollinger_bands}, RSI: {rsi}, MACD: {macd}" + Style.RESET_ALL)
            return 'sell'
        else:
            return 'hold'


        return 'hold' # Default to hold


class MerchantBot(CryptoBot):
    def __init__(self, strategy):
        print(Fore.GREEN + "Initializing MerchantBot..." + Style.RESET_ALL)
        super().__init__(strategy)
        self.start_time = datetime.now()
        self.analyst_bot = AnalystBot(self.strategy)

    @sleep_and_retry
    def get_quantity(self, ticker):
        print(Fore.GREEN + f"Getting quantity for {ticker}..." + Style.RESET_ALL)
        positions = r.get_crypto_positions()
        for position in tqdm(positions):
            if position['currency']['code'] == ticker:
                return float(position['quantity'])
        return 0

    @sleep_and_retry
    async def trade_coin(self, ticker):
        try:
            print(Fore.GREEN + f"Trading {ticker}..." + Style.RESET_ALL)
            signal = await self.analyst_bot.analyze_coin(ticker)
            quantity = self.get_quantity(ticker)
            print(Fore.GREEN + f"[{datetime.now()}] {ticker} quantity: {quantity}" + Style.RESET_ALL)
            elapsed_time = datetime.now() - self.start_time
            if elapsed_time.total_seconds() < 3600 and not ONMODE:
                print(f'Signal is {signal}')
                if quantity > 0 and signal == 'sell':
                    print(Fore.GREEN + f"[PAPER TRADE] Would sell {quantity} of {ticker}" + Style.RESET_ALL)
                elif quantity > 0 and signal == 'buy':
                    print(Fore.GREEN + f"[PAPER TRADE] Would sell {quantity * 0.60} of {ticker}" + Style.RESET_ALL)
                elif quantity == 0 and signal == 'buy':
                    print(Fore.GREEN + f"[PAPER TRADE] Would buy {ticker}" + Style.RESET_ALL)
            elif ONMODE:
                print(f'Signal is {signal}')
                if quantity > 0 and signal == 'sell':
                    r.order_sell_crypto_by_quantity(ticker, quantity, timeInForce='gtc')
                elif quantity > 0 and signal == 'buy':
                    r.order_buy_crypto_by_price(ticker, 0.50, timeInForce='gtc')
                elif quantity == 0 and signal == 'buy':
                    r.order_buy_crypto_by_price(ticker, 1.00, timeInForce='gtc')
            else:
                if quantity > 0 and signal == 'sell':
                    r.order_sell_crypto_by_quantity(ticker, quantity, timeInForce='gtc')
                elif quantity > 0 and signal == 'buy':
                    r.order_buy_crypto_by_price(ticker, 0.50, timeInForce='gtc')
                elif quantity == 0 and signal == 'buy':
                    r.order_buy_crypto_by_price(ticker, 1.00, timeInForce='gtc')
        except Exception as e:
            print(Fore.RED + f"Error trading {ticker}: {e}" + Style.RESET_ALL)
            time.sleep(5)

def load_files():
    print(Fore.WHITE + "Loading files..." + Style.RESET_ALL)
    with open('strategy.json', 'r') as f:
        strategy = json.load(f)
    with open('secrets.json', 'r') as f:
        credentials = json.load(f)
    return strategy, credentials


def login_to_robinhood(credentials):
    print(Fore.WHITE + "Logging in to Robinhood..." + Style.RESET_ALL)
    r.login(credentials['username'], credentials['password'])


def main():
    print(Fore.WHITE + "Starting main function..." + Style.RESET_ALL)
    strategy, credentials = load_files()
    login_to_robinhood(credentials)
    merchant = MerchantBot(strategy)
    tasks = [merchant.trade_coin(currency) for currency in strategy["currencies"]]
    asyncio.run(asyncio.wait(tasks))
    print(Fore.WHITE + "Main function completed." + Style.RESET_ALL)
    return r.get_crypto_positions()

import time

if __name__ == '__main__':
    while True:
        main()
        time.sleep(60*10) # Run every 10 minutes
