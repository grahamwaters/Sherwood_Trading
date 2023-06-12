import asyncio
import logging
# import robin_stocks as r
from robin_stocks import robinhood as r
import numpy as np
import json
import time
import pandas as pd
import datetime
import pytz
import os
import colorama
from colorama import Fore, Back, Style
from icecream import ic
# Read the strategy from a JSON file
with open('strategy.json', 'r') as f:
    strategy = json.load(f)
# Read the credentials from a JSON file
with open('secrets.json', 'r') as f:
    credentials = json.load(f)
    username = credentials['username']
    password = credentials['password']
    # alt_username = credentials['alt_username']
    # alt_password = credentials['alt_password']
    print(f'Logging in as you.. to Robinhood')
    try:
        r.login(username, password)
    except:
        print(f'Logging in as alt.. to Robinhood')
        print(f'deprecated')
        # r.login(alt_username, alt_password)
# Create the bots
class Bot:
    def __init__(self, strategy):
        self.strategy = strategy
        self.analysis_lag_seconds = 30
        self.trade_lag_seconds = 60
    async def sleep(self, duration):
        await asyncio.sleep(duration)
class BankerBot(Bot):
    """
    BankerBot is a bot that manages the user's cash balance, coin positions, and orders on Robinhood.
    :param Bot: The base class for all bots. See `Bot <#bot>`_ for more information.
    :type Bot: class
    """
    def __init__(self, portfolio):
        super().__init__(strategy)
        self.username = username
        self.password = password
        self.cash_balance = None
        self.coin_positions = pd.DataFrame() # TODO: Change to dict?
        self.coin_orders = None
        self.portfolio = portfolio
        # make a file called "master_data.csv" if it doesn't exist
        if not os.path.exists('master_data.csv'):
            pd.DataFrame().to_csv('master_data.csv')
    async def get_account_info(self):
        """
        Get the account information from Robinhood.
        :return: The account information
        :rtype: dict
        :doc-author: Trelent
        """
        # save the account info to a csv file
        account_info = r.account.build_user_profile()
        account_info['timestamp'] = datetime.datetime.now(pytz.timezone('US/Eastern'))
        account_info['cash'] = float(account_info['cash'])
        account_info['crypto_buying_power'] = float(account_info['cash'])
        # account_info['equity'] = float(account_info['equity'])
        # account_info['extended_hours_equity'] = float(account_info['extended_hours_equity'])
        # account_info['extended_hours_market_value'] = float(account_info['extended_hours_market_value'])
        # account_info['market_value'] = float(account_info['market_value'])
        # account_info['withdrawable_amount'] = float(account_info['withdrawable_amount'])
        # account_info['last_core_equity'] = float(account_info['last_core_equity'])
        # account_info['last_core_market_value'] = float(account_info['last_core_market_value'])
        # account_info['last_core_withdrawable_amount'] = float(account_info['last_core_withdrawable_amount'])
        # account_info['portfolio_cash'] = float(account_info['portfolio_cash'])
        # account_info['portfolio_equity'] = float(account_info['portfolio_equity'])
        # account_info['portfolio_previous_close'] = float(account_info['portfolio_previous_close'])
        # account_info['portfolio_withdrawable_amount'] = float(account_info['portfolio_withdrawable_amount'])
        # account_info['total_cash'] = float(account_info['total_cash'])
        # account_info['total_equity'] = float(account_info['total_equity'])
        # account_info['total_extended_hours_equity'] = float(account_info['total_extended_hours_equity'])
        # save the account info to a csv file
        pd.DataFrame(account_info, index=[0]).to_csv('master_data.csv', mode='w', header=False)
        return account_info
    async def get_coin_positions(self):
        """
        Get the coin positions from Robinhood.
        :return: The coin positions
        :rtype: dict
        :doc-author: Trelent
        """
        ic()
        # save the coin positions to a csv file
        coin_positions = r.crypto.get_crypto_positions()
        coin_positions['timestamp'] = datetime.datetime.now(pytz.timezone('US/Eastern'))
        pd.DataFrame(coin_positions).to_csv('master_data.csv', mode='a', header=False)
        return coin_positions
    # update portfolio with new coin positions
    async def update_portfolio(self):
        ic()
        # get the coin positions
        coin_positions = await self.get_coin_positions()
        # get the account info
        account_info = await self.get_account_info()
        # update the portfolio
        self.portfolio.update(coin_positions, account_info)
    # update the portfolio with new coin positions and account info
    async def update_portfolio(self):
        ic()
        # get the coin positions
        coin_positions = await self.get_coin_positions()
        # get the account info
        account_info = await self.get_account_info()
        # update the portfolio
        self.portfolio.update(coin_positions, account_info)
        print(Fore.BLUE + f'Portfolio updated at {datetime.datetime.now(pytz.timezone("US/Eastern"))}')
    # get the coin orders
    async def get_coin_orders(self):
        ic()
        # save the coin orders to a csv file
        coin_orders = r.crypto.get_crypto_orders()
        coin_orders['timestamp'] = datetime.datetime.now(pytz.timezone('US/Eastern'))
        pd.DataFrame(coin_orders).to_csv('master_data.csv', mode='a', header=False)
        return coin_orders
    async def run(self):
        """
        The run function is the main function of your bot. It will be called by the
            framework when it's time to run your bot.
        :param self: Access the attributes and methods of the class
        :return: A coroutine object
        :doc-author: Trelent
        """
        # Log in to Robinhood
        r.login(self.username, self.password)
        while True:
            # Get the account information from Robinhood by asking the BankerBot
            info = await self.get_account_info()
            # Print the cash balance
            # print(f"[BANKER BOT] - Cash balance: {info['cash']}")
            # print it in green
            print(f"[BANKER BOT] - Cash balance: {Fore.GREEN}{info['cash']}{Style.RESET_ALL}")
            # Wait for 60 seconds
            await self.sleep(60)
class AnalystBot(Bot):
    def __init__(self, ticker, strategy):
        super().__init__(strategy)
        self.ticker = ticker
        self.buy_signal_strength = 0
        self.sell_signal_strength = 0
        self.stop_loss = None
        self.trailing_stop_loss = None
        self.purchase_price = None
    async def analyze(self):
        ic()
        while True:
            # Fetch the Bollinger Bands and RSI data
            bollinger_bands = await self.get_bollinger_bands(self.ticker)
            rsi = await self.get_rsi(self.ticker)
            # Calculate the signal strengths
            self.calculate_buy_signal_strength(bollinger_bands, rsi)
            self.calculate_sell_signal_strength(bollinger_bands, rsi)
            # Update the trailing stop loss
            self.update_trailing_stop_loss()
            # Sleep for a while before the next iteration
            await asyncio.sleep(60)
    async def fetch_price_from_api(self, ticker):
        ic()
        ticker = ticker.upper()
        # Fetch the price history
        price_history = r.get_crypto_historicals(ticker, interval='5minute', span='day')
        # Extract the closing prices
        closing_prices = [float(bar['close_price']) for bar in price_history]
        # return the latest closing price
        return closing_prices[-1]
    async def get_price_history(self, ticker):
        ic()
        # Fetch the price history
        price_history = r.crypto.get_crypto_historicals(ticker, interval='5minute', span='day')
        # Extract the closing prices
        closing_prices = [float(bar['close_price']) for bar in price_history]
        # Return the closing prices
        return closing_prices
    async def get_bollinger_bands(self, ticker):
        ic()
        # Fetch the price history
        price_history = await self.get_price_history(ticker)
        # Calculate the moving average
        ma = np.mean(price_history)
        # Calculate the standard deviation
        sd = np.std(price_history)
        # Calculate the upper and lower bands
        upper_band = ma + 2 * sd
        lower_band = ma - 2 * sd
        # Return the Bollinger Bands
        return {'MA': ma, 'BB_up': upper_band, 'BB_down': lower_band}
    async def get_rsi(self, ticker):
        ic()
        # Fetch the price history
        price_history = await self.get_price_history(ticker)
        # Calculate the price differences
        deltas = np.diff(price_history)
        # Initialize the arrays
        seed = deltas[:14]
        up = []
        down = []
        # Separate the price differences into up and down arrays
        for i, delta in enumerate(seed):
            if delta > 0:
                up.append(delta)
            else:
                down.append(delta)
        # Calculate the average gain and loss
        average_gain = sum(up) / 14
        average_loss = abs(sum(down) / 14)
        # Calculate the RSI
        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))
        # Return the RSI
        return rsi
    def calculate_buy_signal_strength(self, bollinger_bands, rsi, price):
        ic()
        # Calculate the Bollinger Band width
        bb_width = bollinger_bands['BB_up'] - bollinger_bands['BB_down']
        # Calculate the distance from the lower band
        distance_from_lower_band = bollinger_bands['BB_down'] - price
        # Calculate the buy signal strength
        self.buy_signal_strength = (bb_width / distance_from_lower_band) * (100 / rsi)
    def calculate_sell_signal_strength(self, bollinger_bands, rsi, price):
        ic()
        # Calculate the Bollinger Band width
        bb_width = bollinger_bands['BB_up'] - bollinger_bands['BB_down']
        # Calculate the distance from the upper band
        distance_from_upper_band = price - bollinger_bands['BB_up']
        # Calculate the sell signal strength
        self.sell_signal_strength = (bb_width / distance_from_upper_band) * rsi
    def update_trailing_stop_loss(self, price):
        ic()
        # If the price is higher than the purchase price, update the trailing stop loss
        if self.purchase_price is None: # If we haven't purchased the stock yet, return
            return
        if price > self.purchase_price:
            self.trailing_stop_loss = price * (1 - self.strategy['trailing_stop_loss'])
    async def run(self):
        ic()
        # Perform analysis and then send signals to the merchant bot which will execute the trades
        while True:
            # Check how much cash we have by checking the bank
            # Fetch the Bollinger Bands
            bollinger_bands = await self.get_bollinger_bands(self.ticker)
            # Fetch the RSI
            rsi = await self.get_rsi(self.ticker)
            # Fetch the current price
            price = await self.fetch_price_from_api(self.ticker)
            # Calculate the buy and sell signal strengths
            self.calculate_buy_signal_strength(bollinger_bands, rsi, price)
            self.calculate_sell_signal_strength(bollinger_bands, rsi, price)
            # Update the trailing stop loss
            self.update_trailing_stop_loss(price)
            # Wait for analysis_lag_seconds before performing the next analysis
            await asyncio.sleep(10)
class MerchantBot(Bot):
    def __init__(self, ticker, analyst_bot, strategy):
        super().__init__(strategy)
        self.ticker = ticker
        self.analyst_bot = analyst_bot
        self.position = None
        self.purchase_price = None
        self.stop_loss = None
        self.trailing_stop_loss = None
    # fetch_price_from_api and get_price are the same function
    async def fetch_price_from_api(self, ticker):
        ic()
        ticker = ticker.upper()
        # Fetch the price history
        price_history = r.get_crypto_historicals(ticker, interval='5minute', span='day')
        # Extract the closing prices
        closing_prices = [float(bar['close_price']) for bar in price_history]
        # return the latest closing price
        return closing_prices[-1]
    async def get_price(self, ticker):
        ic()
        # Fetch the current price
        price = await self.fetch_price_from_api(ticker)  # replace with actual implementation
        return price
    async def ask_analyst_for_update(self):
        # Ask the analyst bot for an update on fetch_price_from_api and sell_signal_strength
        await self.analyst_bot.update()
    async def buy(self, price):
        ic()
        print(Fore.RED + 'BUYING' + Fore.RESET)
        # Calculate the number of shares to buy
        shares = self.strategy['max_buy'] // price
        # Buy the shares
        r.order_buy_crypto_by_price(self.ticker, shares) # buy shares of crypto using robinhood api endpointual
        # Set the purchase price
        self.purchase_price = price
        # Set the stop loss
        self.stop_loss = price * (1 - self.strategy['stop_loss'])
        # Set the trailing stop loss
        self.trailing_stop_loss = price * (1 - self.strategy['trailing_stop_loss'])
        # inform the analyst bot that we have a position and what the purchase price is, so that it can update the trailing stop loss
        self.analyst_bot.position = {'purchase_price': self.purchase_price, 'volume_of_coins': shares}
    async def sell(self, volume_of_coins):
        ic()
        print(Fore.RED + 'SELLING' + Fore.RESET)
        # Sell the shares
        r.order_sell_crypto_by_quantity(self.ticker, volume_of_coins) # sell shares of crypto using robinhood api endpoint
        # inform the analyst bot that it needs to wait a minute and then recalculate the holdings for self.ticker
        self.analyst_bot.position = self.analyst_bot.position - volume_of_coins # todo -- tweak this
        # Reset the position
        self.position = None
        # Reset the purchase price
        self.purchase_price = None
        # Reset the stop loss
        self.stop_loss = None
        # Reset the trailing stop loss
        self.trailing_stop_loss = None
    async def trade(self):
        while True:
            ic()
            # Fetch the current price
            price = await self.get_price(self.ticker)
            # Fetch the volume of coins (of the current position) in the portfolio
            volume_of_coins = self.position['volume_of_coins'] if self.position is not None else 0
            # If there's a strong buy signal and we don't have a position, buy
            if self.analyst_bot.buy_signal_strength >= 2 and self.position is None:
                await self.buy(price)
            # If there's a strong sell signal and wehave a position, sell
            elif self.analyst_bot.sell_signal_strength >= 2 and self.position is not None:
                await self.sell(volume_of_coins)
            # If the price drops below the trailing stop loss, sell
            elif self.position is not None and price < self.analyst_bot.trailing_stop_loss:
                await self.sell(volume_of_coins)
            # Update the trailing stop loss if the price is higher than the purchase price
            elif self.position is not None and price > self.purchase_price:
                self.trailing_stop_loss = max(self.trailing_stop_loss, price * (1 - self.strategy['trailing_stop_loss']))
            # Sleep for a while before the next iteration
            await asyncio.sleep(5)
    async def run(self):
        # Run the trade loop
        await self.trade()
        # Run the ask_analyst_for_update loop
        await self.ask_analyst_for_update()
class MonitorBot(Bot):
    def __init__(self, bots, strategy):
        super().__init__(strategy)
        self.bots = bots
        self.ticker = None
        self.position = None
        self.purchase_price = None
        self.stop_loss = None
        self.trailing_stop_loss = None
        self.buy_signal_strength = None
        self.sell_signal_strength = None
        self.cash = None
    async def monitor(self):
        while True:
            for bot in self.bots:
                if isinstance(bot, AnalystBot):
                    if bot.buy_signal_strength > 2 or bot.sell_signal_strength > 2:
                        # print(f"Warning: AnalystBot for {bot.ticker} has a signal strength greater than 2.")
                        # use colorama to make monitor bot print statements orange using Fore.YELLOW and Style.BRIGHT then reset the color
                        print(Fore.YELLOW + Style.BRIGHT + f"Warning: AnalystBot for {bot.ticker} has a signal strength greater than 2." + Style.RESET_ALL)
                elif isinstance(bot, MerchantBot):
                    if bot.position is not None and bot.trailing_stop_loss > bot.stop_loss:
                        # print(Fore.YELLOW + Style.BRIGHT + f"Warning: AnalystBot for {bot.ticker} has a signal strength greater than 2." + Style.RESET_ALL)
                        print(Fore.YELLOW + Style.BRIGHT + f"Warning: MerchantBot for {bot.ticker} has a trailing stop loss greater than the stop loss." + Style.RESET_ALL)
                elif isinstance(bot, BankerBot):
                    if bot.cash < 0:
                        # print(f"Warning: BankerBot has negative cash.")
                        print(Fore.YELLOW + Style.BRIGHT + f"Warning: BankerBot has negative cash." + Style.RESET_ALL)
                elif isinstance(bot, ScribeBot):
                    pass
                else:
                    pass # todo -- raise an error here
            await asyncio.sleep(30)
    async def run(self):
        # Run the monitor loop
        await self.monitor()
        # use orange print statements for the monitor bot
        print(f"MonitorBot is running.", color="orange") # replace with actual implementation
class ScribeBot(Bot):
    def __init__(self, strategy):
        super().__init__(strategy)
    # if a bot does something then log it
    async def log(self):
        pass
    async def run(self):
        pass
async def main():
    # Import the strategy
    with open('strategy.json') as f:
        strategy = json.load(f) # this strategy will be passed to the bots
    portfolio = strategy['currencies']
    # Create the bots
    ticker = 'DOGE'
    # strategy = strategy[ticker]
    banker_bot = BankerBot(portfolio)
    analyst_bot = AnalystBot(ticker, strategy)
    merchant_bot = MerchantBot(ticker, analyst_bot, strategy)
    monitor_bot = MonitorBot([analyst_bot, merchant_bot], strategy)
    scribe_bot = ScribeBot(strategy)
    # Run the bots
    await asyncio.gather(
        banker_bot.run(),
        analyst_bot.run(),
        merchant_bot.run(),
        monitor_bot.run(),
        scribe_bot.run()
    )
if __name__ == '__main__':
    asyncio.run(main())