import os
import time
import json
import random
from robin_stocks import robinhood as r
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
import yfinance as yf
import pandas as pd
import numpy as np


class TraderClass:
    def __init__(self):
        """
        The __init__ function is called when the class is instantiated.
        It's used to initialize variables and do other setup for the object.


        :param self: Represent the instance of the class
        :return: The object that is created
        :doc-author: Trelent
        """

        with open('secrets.json') as f:
            data = json.load(f)
            self.username = data['username']
            self.password = data['password']
        self.login()
        print(f'Logged in as {self.username}')
        self.update_account_info()
        self.crypto_symbols = r.crypto.get_crypto_currency_pairs()
        self.stocks_list = []
        self.trailing_stop_losses = {}
        # build crypto holdings from profile
        self.holdings = r.get_crypto_positions()
        crypto_holdings = [
            [r.get_crypto_positions()[i]['currency']['code'], float(r.get_crypto_positions()[i]['quantity'])] for i in range(len(r.get_crypto_positions()))
        ]
        self.last_five_prices = {} # todo -- add this to update_account_info
        self.holdings = {crypto_holdings[i][0]: crypto_holdings[i][1] for i in range(len(crypto_holdings))}
        self.crypto_holdings = crypto_holdings
        #!self.buy_prices = {symbol: float(self.holdings[symbol]['average_buy_price']) for symbol in self.holdings}
        self.profile = r.profiles.load_account_profile()
        # build stock holdings
        self.trailing_stop_loss_percent = 0.01
        self.buy_in_percent = 0.01

    def login(self):
        """
        The login function takes in a username and password, and returns an authenticated session.
            The login function is used to authenticate the user with Robinhood's API.
            This allows for the use of other functions that require authentication.

        :param self: Refer to the current instance of the class
        :return: A response object
        :doc-author: Trelent
        """

        r.login(self.username, self.password)
    @sleep_and_retry
    @limits(calls=1, period=600)
    def update_account_info(self):
        print(f'--- building holdings ---')
        self.holdings = r.account.build_holdings() #todo -- holdings is not accurate
        print(f'--- loading account profile ---')
        self.profile = r.profiles.load_account_profile()
        print(f'--- loading ---')
        self.buying_power = float(self.profile['portfolio_cash'])
        self.total_cash = float(self.profile['portfolio_cash']) + float(self.profile['buying_power'])
        self.equity = sum(float(self.holdings[symbol]['quantity']) * float(self.holdings[symbol]['price']) for symbol in self.holdings)
        self.buy_in_percent = 0.01
        self.total_profit = float(self.equity) + self.buying_power - self.total_cash
        self.total_profit_percent = self.total_profit / self.total_cash * 100
        os.environ['TOTAL_CASH'] = str(self.total_cash)

    def get_last_five_prices(self, symbol):
        # gets the last five prices for a symbol
        # todo -- add this to update_account_info
        last_five_prices = []
        historical_data = r.crypto.get_crypto_historicals(symbol, interval='5minute', span='hour', bounds='24_7')
        for i in range(len(historical_data)):
            last_five_prices.append(float(historical_data[i]['close_price']))
        return last_five_prices

    def update_last_five_prices(self):
        # pop the oldest price and add the newest price
        # todo -- add this to update_account_info
        for symbol in self.holdings:
            self.last_five_prices[symbol].pop(0)
            self.last_five_prices[symbol].append(float(r.crypto.get_crypto_quote(symbol)['mark_price']))
    def run_trading_strategy(self):
        """
        The run_trading_strategy function is the main function that runs our trading strategy.
        It loops through all of the symbols in our holdings and checks to see if any of them have hit their trailing stop loss.
        If they have, it executes a sell order for that symbol at market price.
        Then, it loops through all of the symbols in our stocks_list and crypto_symbols lists (which are defined by us) and checks to see if any of them are below their buy-in price (defined by us). If they are, we execute a buy order for that symbol at market price.
        :param self: Represent the instance of the class
        :return: None
        :doc-author: Trelent
        """
        self.trailing_stop_losses = {symbol: TrailingStopLoss(symbol, self) for symbol in self.holdings}
        # symbols = [self.crypto_symbols[i]['symbol'].split('-')[0] for i in range(len(self.crypto_symbols))]
        symbols = ['BTC', 'ETH', 'ADA', 'DOGE', 'MATIC', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'SOL', 'AVAX', 'XLM', 'BCH', 'XTZ']
        # symbols = symbols + self.stocks_list
        # todo -- add stocks in as well
        price_history = pd.DataFrame()







        while True:
            for symbol, trailing_stop_loss in tqdm(self.trailing_stop_losses.items()):
                # for crypto only
                if symbol not in symbols:
                    continue
                tqdm.write(f'Running strategy for {symbol}')
                current_price = float(r.crypto.get_crypto_quote(symbol)['mark_price'])
                trailing_stop_loss.update(current_price)
            for symbol in tqdm(symbols):
                tqdm.write(f'>> Running strategy for {symbol}')
                # for crypto only
                if symbol not in symbols:
                    #buy 1.00 usd if we have enough buying power
                    if self.buying_power > 1.00:
                        self.execute_buy_order(symbol, 1.00)
                        print(f'>> Bought {symbol} at market price')
                        symbols.append(symbol)
                    continue
                historical_data = r.crypto.get_crypto_historicals(symbol, interval='5minute', span='hour', bounds='24_7')
                df = pd.DataFrame(historical_data)
                df['close_price'] = df['close_price'].astype(float)
                df['close_price'] = df['close_price'].values.max()
                df['high_price'] = df['high_price'].astype(float)
                df['high_price'] = df['high_price'].values.max()
                df['low_price'] = df['low_price'].astype(float)
                df['low_price'] = df['low_price'].values.min()
                df['open_price'] = df['open_price'].astype(float)
                df['volume'] = df['volume'].astype(float)
                # check if the latest price is down 1% from the highest price in the last five hours
                # todo -- I don't think this is working
                if float(df['close_price'].iloc[-1]) < float(df['high_price'].max()) * 0.99:
                    # if it is, buy in
                    try:
                        self.execute_buy_order(symbol, self.buy_in_percent)
                        print(f'>> Bought {symbol} at market price')
                    except:
                        print(f'>> Failed to buy {symbol} at market price')

                    # print out the trigger prices for each symbol
                    # for symbol, trailing_stop_loss in self.trailing_stop_losses.items():
                        # print(f'{symbol} trigger price: {trailing_stop_loss}')
            #* Now if the current price is higher than the lowest price seen in the last five checks by 1% or more, we can sell

            # # add the latest price to the price history for each symbol
            # price_history = pd.concat([price_history, pd.Series(df['close_price'].values.max())], axis=1)
            # # if the price history is longer than 5, remove the oldest price
            # if len(price_history) > 5:
            #     price_history.pop(0)
            # # check if the current price is higher than the lowest price seen in the last five checks by 1% or more
            # if price_history[-1]['price'] > min([price['price'] for price in price_history]) * 1.01:
            #     # if it is, sell
            #     self.execute_sell_order(symbol)

            self.update_account_info()
            tqdm.write('Total Cash: ${:.2f}'.format(self.total_cash))
            tqdm.write('Total Profit: ${:.2f} ({:.2f}%)'.format(self.total_profit, self.total_profit_percent))
            tqdm.write('Equity: ${:.2f}'.format(self.equity))
            tqdm.write('Buying Power: ${:.2f}'.format(self.buying_power))
            tqdm.write(f'sleeping for 10 minutes')
            time.sleep(600)
    def execute_buy_order(self, symbol, percent):
        """
        The execute_buy_order function takes in a symbol and a percent of the account's buying power to invest.
        It then calculates the quantity of shares that can be purchased with that amount, and executes an order to buy those shares.

        :param self: Reference the instance of the class
        :param symbol: Specify which cryptocurrency you want to buy
        :param percent: Determine how much of the buying power to use for the order
        :return: The current price of the asset and the quantity that you can buy with your buying power
        :doc-author: Trelent
        """

        current_price = float(r.crypto.get_crypto_quote(symbol)['mark_price'])
        quantity = self.buying_power * percent / current_price
        r.orders.order_buy_crypto_by_price(symbol, quantity)
        self.update_account_info()
    def execute_sell_order(self, symbol, percent):
        """
        The execute_sell_order function takes in a symbol and a percent, and sells that percentage of the quantity of the stock.
        It then updates account info.

        :param self: Make the function a method of the class
        :param symbol: Specify which coin you want to sell
        :param percent: Determine the percentage of holdings to sell
        :return: The order_sell_crypto_by_quantity function
        :doc-author: Trelent
        """
        percent = 1 - percent
        quantity = self.holdings[symbol]['quantity'] * percent
        r.orders.order_sell_crypto_by_quantity(symbol, quantity)
        self.update_account_info()
    @sleep_and_retry
    @limits(calls=1, period=300)
    def get_crypto_data(self, symbol, interval='5minute', span='day'):
        """
        The get_crypto_data function takes in a symbol, interval, and span.
        The function then returns the historical data for that symbol at the given interval and span.

        :param self: Represent the instance of the class
        :param symbol: Specify the crypto currency you want to get data for
        :param interval: Specify the time interval between data points
        :param span: Specify the time period for which we want to get historical data
        :return: A dictionary with the following keys:
        :doc-author: Trelent
        """

        return r.crypto.get_crypto_historicals(symbol, interval=interval, span=span, bounds='24_7')
class TrailingStopLoss:
    def __init__(self, symbol, trader):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the object with all of its instance variables and other things it needs to function properly.

        :param self: Represent the instance of the class
        :param symbol: Identify the stock that is being traded
        :param trader: Access the trader's account balance and to execute trades
        :return: Nothing, it is a constructor
        :doc-author: Trelent
        """
        self.price_history = []
        self.symbol = symbol
        self.trader = trader
        self.stop_loss_price = None
        self.stop_loss_percent = 0.01
        self.buy_in_condition_percent = 0.05
        self.buy_in_percent = 0.01
    def update(self, current_price):
        """
        The update function is called every time the bot receives a new price update.
        It should contain all of your logic for deciding whether to buy or sell, and how much.
        The function takes in two arguments: self (the Bot object) and current_price (a float).
        You can access any of the attributes you defined in __init__ from self, e.g.:

        :param self: Make the class methods aware of other attributes and methods on the same object
        :param current_price: Update the stop loss price
        :return: None
        :doc-author: Trelent
        """
        # if we don't have a stop loss price yet, set it to the current price
        if self.stop_loss_price is None:
            self.stop_loss_price = current_price
            return
        # if the current price is greater than the stop loss price, update the stop loss price
        if current_price > self.stop_loss_price:
            self.stop_loss_price = current_price
            return
        # if the current price is less than the stop loss price, check to see if it's less than the stop loss trigger price
        if current_price < self.stop_loss_price * (1 - self.stop_loss_percent):
            # if it is, sell everything
            self.trader.execute_sell_order(self.symbol, 1)
            # and then set the stop loss price back to None
            self.stop_loss_price = None
            return
        # if the current price is less than the stop loss price, check to see if it's less than the buy in trigger price
        if current_price < self.stop_loss_price * (1 - self.buy_in_condition_percent):
            # if it is, buy in
            self.trader.execute_buy_order(self.symbol, self.buy_in_percent)
            # and then set the stop loss price back to None
            self.stop_loss_price = None
            return


trader = TraderClass()
trader.run_trading_strategy()
