import json
import logging
import pandas as pd
import pandas_ta as ta
import robin_stocks as rstocks
from robin_stocks import robinhood as r
from datetime import datetime
from pytz import timezone
import asyncio
import uuid
import time
import math
import os
from tqdm import tqdm
from colorama import Fore, Back, Style
from icecream import ic
import configparser
import sys
import traceback
import warnings
import alive_progress
from alive_progress import alive_bar
import random
import numpy as np
from time import sleep
from legacy.V5.main import calculate_ta_indicators
#^ load the relevant variables from the .ini credentials file in config/
# [trading]
# coins = BTC, ETH, DOGE, SHIB, ETC, UNI, AAVE, LTC, LINK, COMP, AVAX, XLM, BCH, XTZ
# stop_loss_percent = 0.05
# percent_to_use = 0.8

# [logging]
# verbose_mode = True
# debug_verbose = True
# reset_positions = False
"""
Notes
r.crypto.get_crypto_quote(position['currency']['code'])['mark_price']
use the format above to get the current price of a coin in USD

ordering crypto should use this syntax:
    r.orders.order_crypto(
        symbol = coin,
        amountIn = 'dollars', # or 'quantity'
        side = 'buy',
        quantityOrPrice = buy_cost,#buy_cost / float(df.close.iloc[-1]), # this is the amount of the coin to buy in units of the coin
        limitPrice = df.close.iloc[-1],
        timeInForce = 'gtc',
        jsonify = True
    )

Increments apply to each kind of coin
    r.crypto.get_crypto_info('BTC')['min_order_size'] # this is the minimum amount of BTC that can be bought at a time
# Get crypto positions
positions = r.crypto.get_crypto_positions()
print(f'found {len(positions)} positions.')
for position in tqdm(positions):
    # print(position)
    if position['currency']['code'] == crypto:
        pos_dict = position['currency']
        min_order_size = float(pos_dict['increment'])
        coin_holdings = float(position['quantity_available'])


"""



config = configparser.ConfigParser()
config.read('config/credentials.ini')
coins = config['trading']['coins'].split(', ')
stop_loss_percent = float(config['trading']['stop_loss_percent'])
coins = [coin.strip() for coin in coins]
percent_to_use = float(config['trading']['percent_to_use'])
verbose_mode = config['logging']['verbose_mode']
debug_verbose = config['logging']['debug_verbose']
reset_positions = config['logging']['reset_positions']
minimum_usd_per_position = float(config['trading']['minimum_usd_per_position']) #note: this is the minimum amount of USD that must be invested at any given time in each position. All trades must keep this in mind.
#^ Global Variables
using_trading_function = False # this variable is used to determine if the trading function is currently being used to trade or not. If it is, the trading function will not be called again until it is done trading.
lot_details_list = []

#^ Logging Setup
if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(filename='logs/robinhood.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#^ Robinhood Login Setup
#? rstocks.login(username=config['robinhood']['username'], password=config['robinhood']['password'])
#^ Utility Class
class Utility:
    def __init__(self):
        """
        The Utility class provides utility functions such as getting historical data and checking if it's daytime.
        :doc-author: Trelent
        """
        pass
    async def log_file_size_checker():
        """
        The log_file_size_checker function is an async function that checks the size of the log file and removes lines from the start of the file to maintain a rolling log of 1000 lines.
        :return: None
        :doc-author: Trelent
        """
        while True:
            #ic()
            with open('logs/robinhood.log', 'r') as f:
                lines = f.readlines()
                if len(lines) > 1000: # if the log file is greater than 1000 lines
                    # find how many lines to remove
                    num_lines_to_remove = len(lines) - 1000
                    # remove the first num_lines_to_remove lines
                    with open('logs/robinhood.log', 'w') as f:
                        f.writelines(lines[num_lines_to_remove:])
            await asyncio.sleep(1200)
    def get_last_100_days(self, coin):
        """
        The get_last_100_days function gets the last 100 days of a particular coin's data.
        :param coin: The coin to get data for
        :return: A DataFrame with the last 100 days of coin data
        :doc-author: Trelent
        """
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
        """
        The is_daytime function checks if the current time is between 8 AM and 8 PM.
        :return: True if it's between 8 AM and 8 PM, False otherwise
        :doc-author: Trelent
        """
        current_time = datetime.now(timezone('US/Central'))
        current_hour = current_time.hour
        if current_hour >= 8 and current_hour <= 20:
            return True
        else:
            return False
class Trader:
    """
    The Trader class provides functions for logging into Robinhood, resetting orders, generating trading signals, executing actions based on these signals, updating the buying power, and checking stop loss prices.
    # Detailed Function Descriptions
    1. login_setup: The login_setup function logs into Robinhood using the provided username and password.
    2. resetter: The resetter function cancels all open orders and sells all positions. This function is used to reset the bot.
    3. calculate_ta_indicators:
        The calculate_ta_indicators function calculates different technical indicators and generates trading signals based on these indicators. The indicators are: EMA, MACD, RSI, Williams %R, Stochastic Oscillator, Bollinger Bands, and Parabolic SAR.
        A boolean is generated based on these indicators. If the boolean is True, a buy signal is generated. If the boolean is False, a sell signal is generated. The signals are returned in a DataFrame.
        :param coins: A list of coins to generate signals for
        :return: A DataFrame with the trading signals for each coin
    4. action_module:
        The action_module function executes actions based on the trading signals. If the signal is a buy signal, the coin is bought. If the signal is a sell signal, the coin is sold, if it is owned.
        :param signals: A DataFrame with the trading signals for each coin
        :return: None
    5. buying_power_updater:
        The buying_power_updater function updates the buying power of the Robinhood account (in USD).
        :return: None
    6. stop_loss_checker:
        The stop_loss_checker function checks if the current price of a coin is below the stop loss price. If it is, the coin is sold.
        :param coins: A list of coins to check the stop loss price for
        :return: None
    """
    def __init__(self, username, password):
        """
        The Trader class provides functions for logging into Robinhood, resetting orders, generating trading signals,
        executing actions based on these signals, updating the buying power, and checking stop loss prices.
        :param username: The username for the Robinhood account
        :param password: The password for the Robinhood account
        :doc-author: Trelent
        """
        self.username = username
        self.password = password
        # Set up logging
        self.logger = logging.getLogger('trader')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # Login to Robinhood
        self.login_setup()
        # self.utility_object = Utility()
        # self.log_file_size_checker = self.utility_object.log_file_size_checker() #note: this should be causing an error because of asyncio
        # self.loop = asyncio.get_event_loop()
    def login_setup(self):
        """
        The login_setup function logs into Robinhood using the provided username and password.
        :doc-author: Trelent
        """
        ic()
        try:
            r.login(self.username, self.password)
            self.logger.info('Logged in to Robinhood successfully.')
        except Exception as e:
            self.logger.error(f'Unable to login to Robinhood... {e}')




    def correct_volume(self, volume, coin):
        try:
            while ' ' in coin:
                coin = coin.replace(
                    ' ',''
                )
            min_order_size = float(r.crypto.get_crypto_info(coin)['min_order_size'])
            corrected_volume = math.ceil(volume / min_order_size) * min_order_size
        except Exception as e:
            self.logger.error(f'Unable to correct volume for {coin}... {e}')
            corrected_volume = volume
        return corrected_volume

    def resetter(self):
        """
        The resetter function cancels all open orders and sells all positions.
        :doc-author: Trelent
        """
        ic()
        try:
            open_orders = r.get_all_open_crypto_orders()
            print(Fore.YELLOW + 'Canceling all open orders...' + Style.RESET_ALL)
            for order in tqdm(open_orders):
                r.cancel_crypto_order(order['id'])
            print(Fore.GREEN + 'All open orders cancelled.')
            self.logger.info('All open orders cancelled.' + Style.RESET_ALL)
            crypto_positions = r.get_crypto_positions()
            for position in crypto_positions:
                #note: keep at least minimum_usd_per_position in each position (USD) value not coin value.
                #^ dict_keys(['account_id', 'cost_bases', 'created_at', 'currency', 'currency_pair_id', 'id', 'quantity', 'quantity_available', 'quantity_held', 'quantity_held_for_buy', 'quantity_held_for_sell', 'quantity_staked', 'quantity_transferable', 'updated_at'])
                # Convert 'quantity' and 'direct_cost_basis' to float before performing the comparison
                quantity = float(position['quantity'])
                direct_cost_basis = float(position['cost_bases'][0]['direct_cost_basis'])
                # equity = float(quantity) * float(position['quote']['mark_price'][-1])
                # Compare the result of the multiplication with the minimum threshold
                if quantity * direct_cost_basis < minimum_usd_per_position:
                    # if position is less than minimum_usd_per_position
                    # then skip it
                    continue
                else:
                    ic()
                    # if position is greater than minimum_usd_per_position
                    # then sell it
                    if position['quantity'] == '0.00000000':
                        continue
                    else:
                        try:
                            # get the quote of the coins price currently
                            quote = r.crypto.get_crypto_quote(position['currency']['code'])['mark_price']
                            # sell the coin at the current price
                            quote = float(quote)

                            r.orders.order_sell_crypto_limit(
                                symbol = position['currency']['code'],
                                quantity = position['quantity'],
                                limitPrice = quote,
                                timeInForce = 'gtc'
                            )
                        except Exception as e:
                            self.logger.error(f'Unable to sell {position["quantity"]} {position["currency"]["code"]}... {e}')

                    self.logger.info(f'Sold {position["quantity"]} {position["currency"]["code"]}.')
            self.logger.info('All positions sold.')
        except Exception as e:
            self.logger.error(f'Unable to reset orders and positions... {e}')
    def calculate_ta_indicators(self, coins):
        """
        The calculate_ta_indicators function calculates different technical indicators and generates trading signals based on these indicators. The indicators are: EMA, MACD, RSI, Williams %R, Stochastic Oscillator, Bollinger Bands, and Parabolic SAR.
        A boolean is generated based on these indicators. If the boolean is True, a buy signal is generated. If the boolean is False, a sell signal is generated. The signals are returned in a DataFrame.
        :param coins: A list of coins to generate signals for
        :return: A DataFrame with the trading signals for each coin
        """
        print(Fore.YELLOW + 'Calculating technical indicators...' + Style.RESET_ALL)
        ic()
        try:
            utility = Utility()
            signals_df = pd.DataFrame()
            # read config file
            config = configparser.ConfigParser()
            config.read('config/credentials.ini')
            # get pct_to_buy_with from config file
            pct_to_buy_with = config['trading']['percent_to_use'] #*fixed
            pct_to_buy_with = float(pct_to_buy_with)
            pct_to_buy_per_trade = config['trading']['percent_to_spend_per_trade'] #* fixed
            pct_to_buy_per_trade = float(pct_to_buy_per_trade)
            print(f'Checking Available Buying Power...')
            total_money = float(r.load_account_profile()['crypto_buying_power']) * pct_to_buy_with
            print(f'Available Buying Power: ${total_money}', end='')
            print(f' * {pct_to_buy_with} = ${total_money * pct_to_buy_with}')
            available_money = total_money * pct_to_buy_with
            for coin in coins:
                print(f'Calculating technical indicators for {coin}...')
                df = utility.get_last_100_days(coin)
                df['coin'] = coin #^ add coin name to df
                df['sma'] = df.close.rolling(window=50).mean()
                df['ema'] = df.close.ewm(span=50, adjust=False).mean()
                df['macd_line'], df['signal_line'], df['macd_hist'] = ta.macd(df.close)
                df['rsi'] = ta.rsi(df.close)
                #^ Note: The code below fixes errors in the tas
                df['williams'] = ta.willr(df.high, df.low, df.close) #^ Fixes error in williams_r
                df['stochastic_k'], df['stochastic_d'] = ta.stoch(df.high, df.low, df.close) #^ Fixes error in stoch

                # ['BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0']
                # the above are the column names for the bollinger bands, these must be put into df as columns on the left of the "=" sign then self.bollinger(df) must be put on the right of the "=" sign
                df[['bollinger_l', 'bollinger_m', 'bollinger_u', 'bollinger_b', 'BBP_5_2.0']] = ta.bbands(df['close'], length=5, std=2)

                # df['bollinger_l'], df['bollinger_m'], df['bollinger_u'] = self.bollinger(df) #^ Fixes error in bollinger_bands


                # to prevent errors, cast all columns to float that are indicators
                # df['sma'] = df['sma'].astype(float)
                # df['ema'] = df['ema'].astype(float)
                # df['macd_line'] = df['macd_line'].astype(float)
                # df['signal_line'] = df['signal_line'].astype(float)
                # df['macd_hist'] = df['macd_hist'].astype(float)
                # df['rsi'] = df['rsi'].astype(float)
                # df['williams'] = df['williams'].astype(float)
                # df['stochastic_k'] = df['stochastic_k'].astype(float)
                # df['stochastic_d'] = df['stochastic_d'].astype(float)
                # df['bollinger_l'] = df['bollinger_l'].astype(float)
                # df['bollinger_m'] = df['bollinger_m'].astype(float)
                # df['bollinger_u'] = df['bollinger_u'].astype(float)
                # df['bollinger_b'] = df['bollinger_b'].astype(float)
                # df['BBP_5_2.0'] = df['BBP_5_2.0'].astype(float)


                df['buy_signal'] = ((df.macd_line > df.signal_line) & (df.rsi < 30)) | ((df.stochastic_k > df.stochastic_d) & (df.williams < -80))
                # sell when macd_line crosses below signal_line and rsi is greater than 70
                df['sell_signal'] = ((df.macd_line < df.signal_line) & (df.rsi > 70)) | ((df.stochastic_k < df.stochastic_d) & (df.williams > -20))
                # if the price is greater than the upper bollinger band, the price is going up, so hold all positions and do not sell them yet because the price is going up
                # df['buy_signal'] = df['buy_signal'] & (df.close > df.bollinger_u)
                # also sell when the price is less than the lower bollinger band because this means the price is going down

                #note: this is where you left off 4.30 PM 7/31/2023
                # df['sell_signal'] = df['sell_signal'] | df.close.iloc[-1] < df.bollinger_l.iloc[-1]




                print(Fore.GREEN + f'BUY SIGNAL [{coin}]: {df.buy_signal.iloc[-1]}' + Style.RESET_ALL, Fore.RED + f'SELL SIGNAL [{coin}]: {df.sell_signal.iloc[-1]}' + Style.RESET_ALL)

                #******************** This is Hacking Trades Into Indicators ********************
                if df.buy_signal.iloc[-1] == True: # if buy signal is true
                    # print(f'Buy signal for {coin}: {df.buy_signal.iloc[-1]}', end=' ')
                    # print(f'${available_money} to buy with')
                    # Buy the Greater of these two values:
                    # 1. The pct_to_buy_per_trade of the available_money to buy with (pct_to_buy_per_trade is the percent of the available_money to buy with)
                    # 2. The minimum_usd_per_position (minimum_usd_per_position is the minimum amount of USD to buy with) of the coin
                    # 3. $1.00 USD worth of the coin

                    buy_cost = max(float(available_money) * float(pct_to_buy_per_trade), float(config['trading']['minimum_usd_per_position']), 1.00)
                    buy_cost = float(buy_cost) # cast buy_cost to float
                    available_money = float(available_money) # cast available_money to float
                    # print(f'buy_cost: {buy_cost}')
                    # Now that we know how much to buy, we can submit an order using the r.orders.order_buy_crypto_limit() function
                    # Round the limit price to the nearest cent
                    rounded_limit_price = round(df.close.iloc[-1], 2)

                    buy_order_response = r.orders.order_crypto(
                        symbol = coin,
                        amountIn = 'dollars', # or 'quantity'
                        side = 'buy',
                        quantityOrPrice = buy_cost,#buy_cost / float(df.close.iloc[-1]), # this is the amount of the coin to buy in units of the coin
                        limitPrice = rounded_limit_price,
                        timeInForce = 'gtc',
                        jsonify = True
                    )
                    # self.logger.info(f'Buy order response: {buy_order_response}')
                    # also submit the stop loss order for the coin
                    print(f'Submitting a stop loss order as well at {df.close.iloc[-1] - stop_loss_percent * df.close.iloc[-1]}')
                    sell_order_response = r.orders.order_crypto(
                        symbol = coin,
                        amountIn = 'quantity', # or 'quantity'
                        side = 'sell',
                        quantityOrPrice = self.correct_volume(
                            volume = float(buy_cost / float(df.close.iloc[-1])),
                            coin = coin), # this is the amount of the coin to buy in units of the coin
                        limitPrice = round(df.close.iloc[-1] - stop_loss_percent * df.close.iloc[-1],2), # this is the stop loss price (the price at which the coin will be sold if it drops below this price)
                        timeInForce = 'gtc',
                        jsonify = True
                    )
                    # self.logger.info(f'Sell order response: {sell_order_response}')
                    lot_details = {
                        'coin': coin,
                        'id': uuid.uuid4().hex,
                        'buy_cost': buy_cost,
                        'buy_price': round(df.close.iloc[-1],2),
                        'stop_loss_price': round(df.close.iloc[-1] - stop_loss_percent * df.close.iloc[-1],2),
                        'buy_order_response': buy_order_response
                    }
                    lot_details_list.append(lot_details)
                    # save the lot_details_list to a file
                    with open('lot_details_list.json', 'w') as f:
                        json.dump(lot_details_list, f, indent=4)
                    # Now that we have bought the coin, we need to update the available_money variable
                    available_money -= buy_cost
                    print(Fore.GREEN + f'(+) Bought {buy_cost} worth of {coin} at {df.close.iloc[-1]}\n\t I have ${available_money} left to buy with after purchase.\n\tBuying Options: {float(available_money) * float(pct_to_buy_per_trade)} | {float(config["trading"]["minimum_usd_per_position"])} | $1.00 ' + Fore.RESET)
                elif df.sell_signal.iloc[-1] == True: # if sell signal is true
                    print(f'Sell signal for {coin}: {df.sell_signal.iloc[-1]}')

                    # Get current positions
                    positions = r.get_crypto_positions()
                    coin_volume_owned = 0.0

                    # Check if you have the coin in your positions
                    for position in positions:
                        if position['currency']['code'] == coin:
                            coin_volume_owned = float(position['quantity'])
                            break

                    if coin_volume_owned != 0.0:
                        # Calculate the current value of the coin in USD
                        coin_current_value_usd = coin_volume_owned * float(df.close.iloc[-1])

                        # We can sell if we have more than the minimum_usd_per_position
                        if coin_current_value_usd > float(config['trading']['minimum_usd_per_position']):
                            volume_to_sell_usd = min(coin_current_value_usd - float(config['trading']['minimum_usd_per_position']), available_money * pct_to_buy_per_trade, 1.00)

                            # Submit an order using the r.orders.order_sell_crypto_limit() function
                            r.orders.order_crypto(
                                symbol = coin,
                                amountIn = 'quantity', # in coins
                                side = 'sell',
                                quantityOrPrice = volume_to_sell_usd / float(df.close.iloc[-1]), # this is the amount of the coin to sell in units of the coin
                                limitPrice = df.close.iloc[-1],
                                timeInForce = 'gtc',
                                jsonify = True
                            )
                            # time.sleep(random.randint(1, 5))
                            # cancel all sell orders for the coin now
                            r.orders.cancel_all_crypto_orders(coin)
                    else:
                        print(f"No {coin} positions to sell.")
                else:
                    pass # no buy or sell signal

                #print(f'Buy signal for {coin}: {df.buy_signal.iloc[-1]}')
                #print(f'Sell signal for {coin}: {df.sell_signal.iloc[-1]}')
                signals_df = signals_df.append(df)
            return signals_df
        except Exception as e:
            self.logger.error(f'Unable to generate trading signals... {e}')
            return pd.DataFrame()


    def log_file_size_checker(self):
        # Check the size of the log files (if it has more than 1000 lines, delete the lines from the top of the file until it has 1000 lines)
        try:
            with open('logs/robinhood.log', 'r') as f:
                lines = f.readlines()
                if len(lines) > 1000:
                    with open('logs/robinhood.log', 'w') as f:
                        f.writelines(lines[-1000:])
        except Exception as e:
            self.logger.error(f'Unable to check log file size... {e}')




    def trading_function(self, signals_df):
        """
        The trading_function function takes the trading signals generated by calculate_ta_indicators() and places trades accordingly.
        :param signals_df: A DataFrame with the trading signals for each coin
        :doc-author: Trelent
        """
        try:

            crypto_positions = r.get_crypto_positions()
            cash_available = float(r.load_account_profile()['crypto_buying_power'])
            for index, row in signals_df.iterrows():
                if row['buy_signal']:
                    buying_power = self.update_buying_power()
                    if buying_power > 0:
                        # ic()
                        logging.info(f'would buy {row["coin"]} at {row["close"]}, but commented out for testing.')
                        #r.order.buy_crypto_limit(symbol=row['coin'],
                                                    # quantity = buying_power / row['close'],
                                                    # limitPrice = row['close'],
                                                    # timeInForce = 'gtc')
                        #self.logger.info(f'Bought {row["coin"]} at {row["close"]}.')
                        #buying_power -= float(cash_available)
                if row['sell_signal']:
                    for position in crypto_positions:
                        if position['currency']['code'] == row['coin']:
                            ic()
                            logging.info(f'would sell {row["coin"]} at {row["close"]}, but commented out for testing.')
                            #r.order.sell_crypto_limit(symbol=row['coin'],
                                                        # quantity = position['quantity'],
                                                        # limitPrice = row['close'],
                                                        # timeInForce = 'gtc')
                            # self.logger.info(f'Sold {row["coin"]} at {row["close"]}.')

        except Exception as e:
            self.logger.error(f'Unable to execute trades... {e}')
    def get_total_crypto_dollars(self):
        """
        The get_total_crypto_dollars function calculates the total value of all crypto owned.
        :return: The total value of all crypto owned
        :doc-author: Trelent
        """
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
        """
        The update_buying_power function updates the buying power of the user's account.
        :return: The updated buying power
        :doc-author: Trelent
        """
        try:
            profile_info = r.load_account_profile()
            cash_available = float(profile_info['cash_available_for_withdrawal'])
            # crypto_dollars = self.get_total_crypto_dollars()
            # buying_power = cash_available + crypto_dollars
            buying_power = cash_available
            return buying_power
        except Exception as e:
            self.logger.error(f'Unable to update buying power... {e}')
            return 0
    def check_stop_loss_prices(self, coins, stop_loss_prices):
        """
        The check_stop_loss_prices function checks if the current price is lower than the stop loss price for any owned coin.
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :doc-author: Trelent
        """
        ic()
        try:
            for coin in tqdm(coins):
                current_price = float(r.crypto.get_crypto_quote(coin)['mark_price'])
                if current_price < stop_loss_prices[coin]:
                    crypto_positions = r.get_crypto_positions()
                    for position in crypto_positions:
                        if position['currency']['code'] == coin:
                            r.order.sell_crypto_limit(symbol=coin,
                                                        quantity=position['quantity'],
                                                        limitPrice=current_price,
                                                        timeInForce='gtc')
                            self.logger.info(f'Sold {coin} at {current_price} due to stop loss.')
        except Exception as e:
            self.logger.error(f'Unable to check stop loss prices... {e}')

class Looper:
    def __init__(self, trader: Trader):
        """
        The Looper class provides functions for running asynchronous operations.
        :param trader: An instance of the Trader class
        :doc-author: Trelent
        """
        self.trader = trader
        # Set up logging
        self.logger = logging.getLogger('looper')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    async def run_async_functions(self, loop_count, coins, stop_loss_prices):
        """
        The run_async_functions function is the main function that runs all of the other functions.
        It will run them simultaneously, and it will also keep track of how many times it has looped.
        The loop_count variable is used to determine when to update buying power, which happens every 10 loops.
        :param loop_count: Keep track of how many times the loop has run
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :return: A coroutine object
        :doc-author: Trelent
        """
        try:
            if loop_count % 10 == 0:
                # update buying power every 10 loops
                self.trader.update_buying_power()
            # self.trader.main(coins, stop_loss_prices)
            # run all async functions simultaneously
            # log_file_size_checker included to prevent log file from getting too large
            self.trader.log_file_size_checker()
        except Exception as e:
            self.logger.error(f'Unable to run async functions... {e}')
    async def main_looper(self, coins, stop_loss_prices):
        """
        The main_looper function is the main loop function. It will run every hour and do the following:
            1) Check if there are any open orders that need to be cancelled
            2) Check if there are any positions that need to be sold (if we're in a sell state)
            3) If we're in a buy state, check for new stocks to buy based on our criteria
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :doc-author: Trelent
        """
        loop_count = 0
        while True:
            try:
                # load coins list from config file
                config_file = configparser.ConfigParser()
                config_file.read('config/credentials.ini')
                coins = config_file['trading']['coins'].split(',')
                #^ run the calculate_ta_indicators function to calculate the technical analysis indicators for each coin and buy/sell by the indicators
                trader.calculate_ta_indicators(coins)
                #^ run the check_stop_loss_prices function to check if the current price is lower than the stop loss price for any owned coin
                await self.run_async_functions(
                    loop_count, coins, stop_loss_prices
                    )
                loop_count += 1
                # use alive progress to display a five minute timer
                with alive_bar(300, bar='blocks', spinner='dots_waves2') as foobar:
                    for i in range(300):
                        sleep(1)
                        foobar()
            except Exception as e:
                self.logger.error(f'Error in main loop... {e}')
# run the program
if __name__ == '__main__':
    #^ set stop losses for each coin by multiplying the current price by the stop loss percent (0.05) and subtracting that from the current price (to get the stop loss price).
    trader = Trader(username= config['robinhood']['username'],
                    password= config['robinhood']['password']) #^ create an instance of the Trader class, and log in with the config's username and password
    stop_loss_prices = {coin: float(r.crypto.get_crypto_quote(coin)['mark_price']) - (float(r.crypto.get_crypto_quote(coin)['mark_price']) * stop_loss_percent) for coin in coins}
    print(f'Stop loss prices: {stop_loss_prices}')
    looper = Looper(trader) #^ create an instance of the Looper class (which will run the Trader class)
    asyncio.run(looper.main_looper(coins, stop_loss_prices)) #^ run the main_looper function