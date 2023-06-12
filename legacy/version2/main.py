from robin_stocks import robinhood as r
import pandas as pd
from colorama import Fore, Style
import numpy as np
import time
import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.animation as animation
from ratelimit import limits, sleep_and_retry
import time
def sleep_and_retry(func):
    time.sleep(1)
    return func
import pandas as pd
def create_position_dataframes(data):
    position_dataframes = {}
    for symbol, details in data.items():
        position_df = pd.DataFrame.from_dict(details, orient='index').T
        position_dataframes[symbol] = position_df
    return position_dataframes
class TraderClass:
    def __init__(self):
        # read credentials from file
        with open('secrets.json') as f:
            data = json.load(f)
            username = data['username']
            password = data['password']
        self.username = username
        self.password = password
        self.login = r.login(self.username, self.password)
        self.holdings = None
        self.equity = None
        self.money_invested = None
        self.buying_power = None
        self.update_account_info()
        self.crypto_symbols = r.crypto.get_crypto_currency_pairs()
        # set up the df for the crypto data
        self.crypto_df = pd.DataFrame(columns=['begins_at', 'open_price', 'close_price', 'high_price', 'low_price', 'volume', 'session', 'interpolated', 'symbol', 'RSI', 'MACD', 'Signal', 'Upper_Band', 'Lower_Band', '%K', '%D', 'MA50', 'MA200', 'Price', 'equity', 'buying_power'])
        self.crypto_holdings = r.crypto.get_crypto_positions()
    #* Account Information Functions
    @sleep_and_retry
    def update_account_info(self):
        self.holdings = r.account.build_holdings()
        self.profile = r.profiles.load_account_profile()
        self.equity = 0
        self.total_cash = float(self.profile['portfolio_cash']) + float(self.profile['buying_power'])
        self.buying_power = float(self.profile['crypto_buying_power'])
        # to get total equity go through self.holdings and multiply the quantity by the current price of the stock or crypto and add it to the equity
        total_equity = [
            float(self.holdings[symbol]['quantity']) * float(self.holdings[symbol]['price']) for symbol in self.holdings
        ]
        avg_profit = [
            float(self.holdings[symbol]['average_buy_price']) - float(self.holdings[symbol]['price']) for symbol in self.holdings
        ]
        self.equity = sum(total_equity)
        self.total_cash = self.equity + self.buying_power
        self.total_profit = self.equity + self.buying_power - self.total_cash
        self.total_profit_percent = self.total_profit / self.total_cash * 100
        print(Fore.GREEN + "Account information updated." + Style.RESET_ALL)
        print(Fore.GREEN + "Equity: " + Style.RESET_ALL + str(self.equity))
        print(Fore.GREEN + "Buying Power: " + Style.RESET_ALL + str(self.buying_power))
        print(Fore.GREEN + "Total Cash: " + Style.RESET_ALL + str(self.total_cash))
        print(Fore.GREEN + "Total Profit: " + Style.RESET_ALL + str(self.total_profit))
        print(Fore.GREEN + "Total Profit Percent: " + Style.RESET_ALL + str(self.total_profit_percent) + "%")
        self.positions_df = create_position_dataframes(self.holdings)
    #* Technical Indicator Functions
    def calculate_RSI(self, df, window_length=14):
        delta = df['close_price'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up1 = up.ewm(span=window_length).mean()
        roll_down1 = down.abs().ewm(span=window_length).mean()
        RS = roll_up1 / roll_down1
        RSI = 100.0 - (100.0 / (1.0 + RS))
        return RSI
    def calculate_MACD(self, df, short_window=12, long_window=26):
        short_ema = df['close_price'].ewm(span=short_window, adjust=False).mean()
        long_ema = df['close_price'].ewm(span=long_window, adjust=False).mean()
        MACD = short_ema - long_ema
        signal = MACD.ewm(span=9, adjust=False).mean()
        return MACD, signal
    def calculate_Bollinger_Bands(self, df, window_length=20, num_of_std=2):
        rolling_mean = df['close_price'].rolling(window=window_length).mean()
        rolling_std = df['close_price'].rolling(window=window_length).std()
        upper_band = rolling_mean + (rolling_std * num_of_std)
        lower_band = rolling_mean - (rolling_std * num_of_std)
        return upper_band, lower_band
    def calculate_Stochastic_Oscillator(self, df, window_length=14):
        low_min = df['low_price'].rolling(window=window_length).min()
        high_max = df['high_price'].rolling(window=window_length).max()
        k_percent = (df['close_price'] - low_min) / (high_max - low_min) * 100
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent
    def calculate_Moving_Average(self, df, window_length=50):
        moving_average = df['close_price'].rolling(window=window_length).mean()
        return moving_average
    def calculate_technical_indicators(self, symbol):
        # Fetch historical data
        historical_data = r.crypto.get_crypto_historicals(
            symbol, interval='day', span='year', bounds='24_7', info=None)
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        # Calculate technical indicators and convert the df values to numeric before calculating
        df['open_price'] = pd.to_numeric(df['open_price'])
        df['close_price'] = pd.to_numeric(df['close_price'])
        df['high_price'] = pd.to_numeric(df['high_price'])
        df['low_price'] = pd.to_numeric(df['low_price'])
        df['Volume'] = pd.to_numeric(df['volume'])
        df['RSI'] = self.calculate_RSI(df)
        df['MACD'], df['Signal'] = self.calculate_MACD(df)
        df['Upper_Band'], df['Lower_Band'] = self.calculate_Bollinger_Bands(df)
        df['%K'], df['%D'] = self.calculate_Stochastic_Oscillator(df)
        df['MA50'] = self.calculate_Moving_Average(df, window_length=50)
        df['MA200'] = self.calculate_Moving_Average(df, window_length=200)
        df['Price'] = df['close_price']
        # Drop NaN values
        df = df.dropna()
        # Convert values to datetime
        df['begins_at'] = pd.to_datetime(df['begins_at'], format="%Y-%m-%dT%H:%M:%SZ")
        print(Fore.GREEN + "Technical indicators calculated." + Style.RESET_ALL)
        return df
    #* Generating Signals
    def generate_buy_signal(self, df):
        # Initialize buy signal strength
        buy_signal_strength = 0
        # Check RSI for oversold condition
        if df['RSI'].iloc[-1] < 30:
            buy_signal_strength += 1
        # Check MACD for bullish crossover
        if df['MACD'].iloc[-1] > df['Signal'].iloc[-1]:
            buy_signal_strength += 1
        # Check Bollinger Bands for price breakout above the upper band
        if df['Price'].iloc[-1] > df['Upper_Band'].iloc[-1]:
            buy_signal_strength += 1
        # Check Stochastic Oscillator for bullish crossover
        if df['%K'].iloc[-1] > df['%D'].iloc[-1]:
            buy_signal_strength += 1
        # Check Moving Average for golden cross
        if df['MA50'].iloc[-1] > df['MA200'].iloc[-1]:
            buy_signal_strength += 1
        # Check Volume for high trading volume
        if df['Volume'].iloc[-1] > df['Volume'].rolling(window=10).mean().iloc[-1]:
            buy_signal_strength += 1
        # Check Price for higher lows and higher highs
        if df['Price'].iloc[-1] > df['Price'].iloc[-2] and df['Price'].iloc[-2] > df['Price'].iloc[-3]:
            buy_signal_strength += 1
        # Put equity in the df
        df['equity'] = self.equity
        print(Fore.GREEN + "Buy signal generated with strength: " + str(buy_signal_strength) + Style.RESET_ALL)
        return buy_signal_strength
    def generate_sell_signal(self, df):
        # Initialize sell signal strength
        sell_signal_strength = 0
        # Check RSI for overbought condition
        if df['RSI'].iloc[-1] > 70:
            sell_signal_strength += 1
        # Check MACD for bearish crossover
        if df['MACD'].iloc[-1] < df['Signal'].iloc[-1]:
            sell_signal_strength += 1
        # Check Bollinger Bands for price breakout below the lower band
        if df['Price'].iloc[-1] < df['Lower_Band'].iloc[-1]:
            sell_signal_strength += 1
        # Check Stochastic Oscillator for bearish crossover
        if df['%K'].iloc[-1] < df['%D'].iloc[-1]:
            sell_signal_strength += 1
        # Check Moving Average for death cross
        if df['MA50'].iloc[-1] < df['MA200'].iloc[-1]:
            sell_signal_strength += 1
        # Check Volume for high trading volume
        if df['Volume'].iloc[-1] > df['Volume'].rolling(window=10).mean().iloc[-1]:
            sell_signal_strength += 1
        # Check Price for lower highs and lower lows
        if df['Price'].iloc[-1] < df['Price'].iloc[-2] and df['Price'].iloc[-2] < df['Price'].iloc[-3]:
            sell_signal_strength += 1
        # use trailing_stop_loss to calculate the stop loss price and see if the current price is below the stop loss price of 1% below the last high price (the highest price since we bought the stock)
        self.trailing_stop_loss = df['Price'].rolling(window=3).max().iloc[-1] * 0.99 # 1% below the last high price
        if df['Price'].iloc[-1] < self.trailing_stop_loss:
            sell_signal_strength += 1
        print(Fore.GREEN + "Sell signal generated with strength: " + str(sell_signal_strength) + Style.RESET_ALL)
        return sell_signal_strength



    #* Order Execution Functions
    @limits(calls=1, period=5)
    def execute_buy_order(self, symbol, buy_signal_strength):
        # Calculate the amount to buy and if we have enough buying power to execute the order. Remember to only play with 60% of your buying power to build in a margin of safety.
        amount_to_buy = max(1.00, 0.6 * float(self.buying_power)) * buy_signal_strength #   1.00 is the minimum order amount
        print(Fore.GREEN + "Amount to buy: $" + str(amount_to_buy) + Style.RESET_ALL)
        if amount_to_buy > self.equity:
            amount_to_buy = self.equity
        # check for existing orders and cancel them
        orders = r.get_all_open_crypto_orders()
        for order in orders:
            if order['side'] == 'buy' and order['currency_pair_id'] == symbol:
                r.cancel_crypto_order(order['id'])
                print(Fore.RED + "\tCancelled existing buy order for " + symbol + Style.RESET_ALL)
        # Execute the buy order
        r.order_buy_crypto_by_price(symbol, amount_to_buy)
        print(Fore.GREEN + "Buy order executed for " + symbol + " with amount: $" + str(amount_to_buy) + Style.RESET_ALL)
    @limits(calls=1, period=5)
    def execute_sell_order(self, symbol, sell_signal_strength):
        # Calculate the amount to sell
        amount_to_sell = sell_signal_strength * float(self.holdings[symbol]['quantity'])
        # Execute the sell order
        r.order_sell_crypto_by_quantity(symbol, amount_to_sell)
        print(Fore.GREEN + "Sell order executed for " + symbol + " with quantity: " + str(amount_to_sell) + Style.RESET_ALL)
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
    # Try making a function that implements a 1% trailing stop loss by canceling s
    def trailing_stop_loss(self, symbol, current_price):
        """
        The trailing_stop_loss function is used to sell a crypto currency when the price drops below 99% of the highest price it has reached since purchase.
            This function takes in two arguments:
                1) self - this is an instance of the Robinhood class that contains all relevant information about your holdings and account.  It also contains methods for executing trades, etc.
                2) symbol - this is a string containing the ticker symbol for which you want to execute a trailing stop loss order (e.g., 'BTC-USD')
        :param self: Represent the instance of the class
        :param symbol: Identify the symbol of the currency pair
        :param current_price: Update the high price for the symbol
        :return: A sell order for the symbol if the current_price is less than 1% of the high price
        :doc-author: Trelent
        """
        if current_price > self.holdings[symbol]['high']:
            self.holdings[symbol]['high'] = current_price # update the high price for the symbol
        if current_price < self.holdings[symbol]['high'] * 0.99:
            # cancel the sell order
            orders = r.get_all_open_crypto_orders()
            for order in orders:
                if order['side'] == 'sell' and order['currency_pair_id'] == symbol:
                    r.cancel_crypto_order(order['id'])
                    print(Fore.RED + "\tCancelled existing sell order for " + symbol + Style.RESET_ALL)
            # Execute the sell order
            r.order_sell_crypto_by_quantity(symbol, self.holdings[symbol]['quantity'])
            print(Fore.GREEN + "Sell order executed for " + symbol + " with quantity: " + str(self.holdings[symbol]['quantity']) + Style.RESET_ALL)
            # reset the high price
            self.holdings[symbol]['high'] = 0
    #* Main Loop
    def main_loop(self, symbol):
        while True:
            # Define the Cryptocurrencies we want to trade
            currenciesList = [
                'BTC',
                'ETH'
            ]
            # reset our crypto holdings
            df_holds = pd.DataFrame(columns=['symbol', 'quantity'])
            # fill the dataframe with our current holdings (crypto)
            # check self.crypto_holdings for each currency in currenciesList
            for currency in currenciesList:

            # set the index to the symbol
            df_holds.set_index('symbol', inplace=True)
            # convert the quantity column to a float
            df_holds['quantity'] = df_holds['quantity'].astype(float)
            # update the holdings
            self.crypto_holdings = df_holds.to_dict('index')
            # Define the time interval for the data
            interval = '30 Minutes'
            # Define the number of days of data to retrieve
            days = 7
            # Define the number of minutes to wait between each iteration of the loop
            minutes = 10
            # loop through each currency
            for currency in currenciesList:
                # get the data for the currency
                df = self.get_crypto_data(currency, interval, days)
                # calculate the indicators
                df = self.calculate_indicators(df)
                # get the buy and sell signals
                buy_signal_strength, sell_signal_strength = self.get_signals(df)
                if buy_signal_strength > 0:
                    self.execute_buy_order(currency, buy_signal_strength)
                if sell_signal_strength > 0:
                    self.execute_sell_order(currency, sell_signal_strength)
                # wait for the specified number of minutes
                time.sleep(minutes * 60)
trader = TraderClass()
# run for 1 hour on BTC
trader.main_loop('BTC')
