import json
import pandas as pd
from datetime import datetime
from pytz import timezone
from robin_stocks import robinhood as r
from colorama import Fore, Style
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry
import time
import requests

ONMODE = True

class CryptoBot:
    def __init__(self, strategy):
        print(Fore.WHITE + "Initializing CryptoBot..." + Style.RESET_ALL)
        self.strategy = strategy
        self.portfolio = pd.DataFrame()
        self.coin_positions = pd.DataFrame()
        self.current_price = None
        self.current_holdings = None
        self.last_check = None
        self.buy_signal_strength = 0
        self.sell_signal_strength = 0
        # login to robinhood
        # read secrets.json
        with open('secrets.json', 'r') as f:
            credentials = json.load(f)
        r.login(credentials['username'], credentials['password'])
        print(Fore.GREEN + "Logged in to Robinhood!" + Style.RESET_ALL)
    @sleep_and_retry
    @limits(calls=1, period=5)  # Place your API rate limit here
    def get_account_info(self):
        print(Fore.WHITE + "Getting account info..." + Style.RESET_ALL)
        accountinfo = r.build_user_profile()
        accountinfo['timestamp'] = datetime.now(timezone('US/Eastern'))
        accountinfo['cash'] = float(accountinfo['cash'])
        accountinfo['crypto_buying_power'] = float(accountinfo['cash'])
        return accountinfo

    @sleep_and_retry
    @limits(calls=1, period=5)  # Place your API rate limit here
    def get_coin_positions(self):
        print(Fore.WHITE + "Getting coin positions..." + Style.RESET_ALL)
        coin_positions = pd.DataFrame(r.get_crypto_positions())
        coin_positions['quantity'] = float(coin_positions['quantity'][0])
        return coin_positions

    @sleep_and_retry
    @limits(calls=1, period=5)  # Place your API rate limit here
    def fill_portfolio(self):
        print(Fore.WHITE + "Filling portfolio..." + Style.RESET_ALL)
        self.portfolio = self.get_account_info()
        self.coin_positions = self.get_coin_positions()

    @limits(calls=1, period=5)  # Place your API rate limit here
    def get_price_history(self, ticker: str, interval='5minute', span='day') -> list:
        print(Fore.WHITE + f"Getting price history for {ticker}..." + Style.RESET_ALL)
        # use requests to get the price history
        # https://nummus.robinhood.com/currency_pairs/
        url = f"https://api.robinhood.com/marketdata/forex/historicals/{ticker}/?interval={interval}&span={span}&bounds=regular"
        response = requests.get(url)
        price_history = response.json()['data_points']
        return [float(bar['close_price']) for bar in price_history]
        # price_history = r.get_crypto_historicals(ticker, interval=interval, span=span)
        # return [float(bar['close_price']) for bar in price_history]

    def sleep(self, duration):
        print(Fore.WHITE + f"Sleeping for {duration} seconds..." + Style.RESET_ALL)
        time.sleep(duration)


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

    def get_current_holdings(self):
        print(Fore.YELLOW + "Updating current holdings..." + Style.RESET_ALL)
        self.current_holdings = self.get_account_info()
        return self.current_holdings

    def calculate_moving_average(self, ticker):
        print(Fore.YELLOW + f"Calculating moving average for {ticker}..." + Style.RESET_ALL)
        try:
            price_history = self.get_price_history(ticker)
            self.current_price = price_history[-1] # Set the current price
            self.last_check = datetime.now() # Set the last check time
            ma = sum(price_history[-50:]) / 50  # Calculate moving average for the last 50 data points
        except Exception as e:
            print(Fore.RED + f"Error calculating moving average for {ticker}: {e}" + Style.RESET_ALL)
            # if the error is too many requests, sleep for 5 minutes
            if "Too many requests" in str(e) or\
                    "too many calls" in str(e):
                self.sleep(300)
            ma = None
        return ma


if __name__ == "__main__":
    config = {
        "currencies": ["BTC", "ETH", "DOGE", "SOL", "LINK", "ADA"],
        "botConfiguration": {
            "tradeInterval": 5,  # in minutes
            "stopLoss": 0.05,  # 5% stop loss
            "maximumInvestment": 0.60,  # only ever invest 60% of the total cash in the account, if the total of invested dips beneath 35% of the total cash (meaning a stop loss will be triggered), then the bot will invest again
            "trailingStopLoss": 0.01  # 1% trailing stop loss
        }
    }
    bot = AnalystBot(config)
    bot.fill_portfolio()

    # while True:
    #     for currency in bot.currencies:
    #         moving_average = bot.calculate_moving_average(currency)
    #         print(f"Moving Average for {currency}: {moving_average}")
    #     bot.sleep(config["botConfiguration"]["tradeInterval"] * 60)

    # just show me the current holdings
    print(bot.get_current_holdings())
    # create a pandas df of the current holdings

    holdings = pd.DataFrame(bot.get_current_holdings())
