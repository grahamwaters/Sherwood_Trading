from robin_stocks import robinhood as r
import pandas as pd
from colorama import Fore, Style
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from ratelimit import limits, sleep_and_retry
import datetime
import os
# All I want is a trailing stop loss of 1% that I can set and forget
# I want to be able to set a trailing stop loss of 1% on all my positions, then have it automatically update as the price goes up, but not down.
# then if the price goes up 1%, the stop loss goes up 1% as well
# if the price goes down, the stop loss stays the same, and if the price drops below the stop loss, it sells the position.
# if the position has been sold (we have none) and the price is down 5% in the last 5 hours, buy back in with 1% of our buying power and set the stop loss to 1% below the current price as usual, repeat.
class TraderClass:
    def __init__(self):
        # read credentials from file
        with open('secrets.json') as f:
            data = json.load(f)
            username = data['username']
            password = data['password']
        self.username = username
        self.password = password
        self.login()
        self.update_account_info()
        self.crypto_symbols = r.crypto.get_crypto_currency_pairs()
        # set up the df for the crypto data
        self.crypto_df = pd.DataFrame(columns=['begins_at', 'open_price', 'close_price', 'high_price', 'low_price', 'volume', 'session', 'interpolated', 'symbol', 'RSI', 'MACD', 'Signal', 'Upper_Band', 'Lower_Band', '%K', '%D', 'MA50', 'MA200', 'Price', 'equity', 'buying_power'])
        self.crypto_data_cache = {}
        self.crypto_data_cache_time = {}
        self.equity = 0
        profile = r.profiles.load_account_profile()
        self.profile = profile
        self.buying_power = float(profile['crypto_buying_power'])
        self.money_invested = float(profile['portfolio_cash'])
        self.total_cash = 0
        self.trailing_stop_loss_percent = 0.01
        self.buy_in_percent = 0.01
        self.total_profit = 0
        self.total_profit_percent = 0
        self.trailing_stop_losses = {}
        # get all positions from  r.profiles.load_account_profile() object already saved as 'self.profile'
        self.positions = self.profile['positions'] # this is a list of dictionaries
        self.total_profit = float(self.equity) + float(self.buying_power) - float(self.total_cash)
        self.total_profit_percent = 0 # this is how much profit has been made since the program started
        # self.positions_df = None
        self.trailing_stop_losses = {} # this is a dictionary of trailing stop losses, the key is the symbol, the value is the TrailingStopLoss object
        self.trailing_stop_loss_percent = 0.01  # this is the percent that the stop loss will trail behind the current price
        self.buy_in_percent = 0.01  # this is the percent of our buying power that we will buy in with if we don't have a position
    def login(self):
        r.login(self.username, self.password)
    @sleep_and_retry
    def update_account_info(self):
        self.holdings = r.account.build_holdings()
        self.profile = r.profiles.load_account_profile()
        self.equity = 0
        self.total_cash = float(self.profile['portfolio_cash']) + float(self.profile['buying_power'])
        self.buying_power = float(self.profile['crypto_buying_power'])
        self.buying_power = float(self.profile['portfolio_cash'])
        # to get total equity go through self.holdings and multiply the quantity by the current price of the stock or crypto and add it to the equity
        total_equity = [
            float(self.holdings[symbol]['quantity']) * float(self.holdings[symbol]['price']) for symbol in self.holdings
        ]
        avg_profit = [
            float(self.holdings[symbol]['average_buy_price']) - float(self.holdings[symbol]['price']) for symbol in self.holdings
        ]
        self.equity = sum(total_equity)
        self.total_cash = self.equity + float(self.buying_power)
        self.total_profit = float(self.equity) + self.buying_power - self.total_cash
        self.total_profit_percent = self.total_profit / self.total_cash * 100
        print(Fore.GREEN + "Account information updated." + Style.RESET_ALL)
        print(Fore.GREEN + "Equity: " + Style.RESET_ALL + str(self.equity))
        print(Fore.GREEN + "Buying Power: " + Style.RESET_ALL + str(self.buying_power))
        print(Fore.GREEN + "Total Cash: " + Style.RESET_ALL + str(self.total_cash))
        # save the total cash to an environment variable
        os.environ['TOTAL_CASH'] = str(self.total_cash)
        print(Fore.GREEN + "Total Profit: " + Style.RESET_ALL + str(self.total_profit))
        print(Fore.GREEN + "Total Profit Percent: " + Style.RESET_ALL + str(self.total_profit_percent) + "%")
        # self.positions_df.to_csv('positions.csv')
    def get_accurate_gains(self):
        """
        The get_accurate_gains function is a function that calculates the total money invested,
        the total equity, and the net worth increase due to dividends and other gains. It does this by
        using Robinhood's API to get all of the bank transactions (deposits/withdrawals), card transactions (debits),
        and portfolio profile data. The deposits are added together with any reversal fees from failed deposits. Then, withdrawals are subtracted from this sum along with debits made on your debit card.
        :return: The total money invested, the total equity,
        :doc-author: Trelent
        """
        profileData = r.load_portfolio_profile()
        allTransactions = r.get_bank_transfers()
        cardTransactions= r.get_card_transactions()
        deposits = sum(float(x['amount']) for x in allTransactions if (x['direction'] == 'deposit') and (x['state'] == 'completed'))
        withdrawals = sum(float(x['amount']) for x in allTransactions if (x['direction'] == 'withdraw') and (x['state'] == 'completed'))
        debits = sum(float(x['amount']['amount']) for x in cardTransactions if (x['direction'] == 'debit' and (x['transaction_type'] == 'settled')))
        reversal_fees = sum(float(x['fees']) for x in allTransactions if (x['direction'] == 'deposit') and (x['state'] == 'reversed'))
        money_invested = deposits + reversal_fees - (withdrawals - debits)
        dividends = r.get_total_dividends()
        percentDividend = dividends/money_invested*100
        equity = float(profileData['extended_hours_equity'])
        totalGainMinusDividends = equity - dividends - money_invested
        percentGain = totalGainMinusDividends/money_invested*100
        print("The total money invested is {:.2f}".format(money_invested))
        print("The total equity is {:.2f}".format(equity))
        print("The net worth has increased {:0.2}% due to dividends that amount to {:0.2f}".format(percentDividend, dividends))
        print("The net worth has increased {:0.3}% due to other gains that amount to {:0.2f}".format(percentGain, totalGainMinusDividends))
        buying_power =  float(profileData['withdrawable_amount'])
        equity = float(profileData['extended_hours_equity'])
        # update the equity and buying power
        self.equity = equity
        self.buying_power = buying_power
        self.money_invested = money_invested
    def run_trading_strategy(self):
        self.trailing_stop_losses = {symbol: TrailingStopLoss(symbol, self) for symbol in self.holdings}
        while True:
            print(f'Total Profit: {self.total_profit_percent}%')
            for symbol, trailing_stop_loss in self.trailing_stop_losses.items():
                if symbol not in ['BTC', 'ETH', 'LTC', 'SOL', 'BSV',
                                    'DOGE', 'ETC', 'BCH', 'ADA', 'XLM',
                                    'LINK', 'XRP', 'DOT', 'UNI', 'AAVE',
                                    'LUNA', 'MATIC']:
                    # this is a stock
                    current_price = float(r.stocks.get_latest_price(symbol)[0]) # get the latest price of the stock
                else:
                    # this is a crypto
                    current_price = float(r.crypto.get_crypto_quote(symbol)['mark_price'])
                # if the current price is less than the stop loss price, sell using a market order, then watch for the RSI to go below 30 and buy back in
                # todo -- add RSI logic in later
                trailing_stop_loss.update(current_price)
            # repeat every ten minutes (600 seconds)
            print('Sleeping for 10 minutes')
            time.sleep(600)
    def execute_buy_order(self, symbol, percent):
        # buy in at the current price
        current_price = float(r.crypto.get_crypto_quote(symbol)['mark_price'])
        quantity = self.buying_power * percent / current_price # buy in with 1% of our buying power
        print(f'Buying {quantity} of {symbol} at {current_price} which costs {quantity * current_price}')
        r.orders.order_buy_crypto_by_price(symbol, quantity)
        self.refresh_data()
    def execute_sell_order(self, symbol, percent):
        # sell all of our holdings
        quantity = self.holdings[symbol]['quantity'] * percent
        print(f'Selling {quantity} of {symbol}')
        r.orders.order_sell_crypto_by_quantity(symbol, quantity)
        self.refresh_data()
    @sleep_and_retry
    def get_crypto_data(self, symbol, interval='5minute', span='day'):
        # check if the data is in the cache
        if symbol in self.crypto_data_cache:
            # check if the data is stale
            if self.crypto_data_cache[symbol]['time'] > datetime.datetime.now() - datetime.timedelta(minutes=5):
                return self.crypto_data_cache[symbol]['data']
        # if not, get the data from robinhood
        data = r.crypto.get_crypto_historicals(symbol, interval=interval, span=span, bounds='24_7')
        # add the data to the cache
        self.crypto_data_cache[symbol] = {'data': data, 'time': datetime.datetime.now()}
        return data
    def cancel_all_orders(self, symbol):
        # cancel all orders for a given symbol
        # if not logged in raise an error
        if self.buying_power is None:
            raise Exception('Not logged in')
        orders = self.trader.get_all_open_crypto_orders()
        for order in orders:
            if order['symbol'] == symbol:
                self.trader.cancel_crypto_order(order['id'])
    def update_trailing_stop_loss_orders(self, current_price):
        # first cancel any existing orders
        # check if we have a position
        if float(self.trader.holdings[self.symbol]['quantity']) > 0:
            # if we have a position, update the stop loss price
            if self.stop_loss_price is None or current_price > self.stop_loss_price:
                self.stop_loss_price = current_price * (1 - self.stop_loss_percent)
            # if the price is above the stop loss price, update the stop loss price
            elif current_price < self.stop_loss_price:
                self.trader.execute_sell_order(self.symbol, 1)
                self.stop_loss_price = None
        # if we don't have a position, check if we should buy in
        else:
            historical_data = self.trader.get_crypto_data(self.symbol, interval='hour', span='5hour')
            # if the price is down 5% in the last 5 hours, buy in
            if historical_data[-1]['close_price'] < historical_data[0]['close_price'] * (1 - self.buy_in_condition_percent):
                # buy in for 1% of our buying power in USD
                self.trader.execute_buy_order(self.symbol, self.buy_in_percent)
                self.stop_loss_price = current_price * (1 - self.stop_loss_percent)
import random
trader = TraderClass()
print(trader.holdings)
time.sleep(random.randint(1, 5))
print(trader.profile)
print(f'Equity: {trader.equity}')
port_cash = trader.profile['portfolio_cash']
os.environ['PORTFOLIO_CASH'] = port_cash
os.environ['BUYING_POWER'] = trader.profile['buying_power']
print(f"Total Cash: {float(port_cash) + trader.total_profit}")
print(f'Total Profit: {trader.total_profit}')
print(f'Total Profit Percent: {trader.total_profit_percent}')
print(f'Running trading strategy...')
trader.run_trading_strategy()
