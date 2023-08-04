import asyncio
from robin_stocks import robinhood as r
import numpy as np
import pandas as pd
import configparser
import logging
import time
import sys
import os
import datetime
from datetime import datetime
from datetime import timedelta
from random import randint, uniform

config = configparser.ConfigParser()
config.read('config/credentials.ini')

coins = [coin.strip() for coin in config['trading']['coins'].split(', ')]
stop_loss_percent = float(config['trading']['stop_loss_percent'])
percent_to_use = float(config['trading']['percent_to_use'])
verbose_mode = config['logging'].getboolean('verbose_mode')
debug_verbose = config['logging'].getboolean('debug_verbose')
reset_positions = config['logging'].getboolean('reset_positions')
minimum_usd_per_position = float(config['trading']['minimum_usd_per_position'])
pct_to_buy_with = float(config['trading']['percent_to_use'])
pct_to_buy_per_trade = float(config['trading']['percent_to_spend_per_trade'])
username = config['robinhood']['username']
password = config['robinhood']['password']
login = r.login(username, password)
print(f"Logged in as {username}")

print(f'coins: {coins}')
print(f'stop_loss_percent: {stop_loss_percent}')
print(f'=' * 20)

logging.basicConfig(level=logging.INFO)

# Initialize paper trading variables
cash = 10000.0  # Starting cash
portfolio = {coin: 0 for coin in coins}  # Starting portfolio
trade_history = []  # Trade history
pnl = 0  # Profit and loss
asset_value = cash  # Total asset value

async def get_current_price(symbol):
    current_price = r.crypto.get_crypto_quote(symbol, info='mark_price')
    return float(current_price)

async def place_order(symbol, order_type, price, stop_loss=None, take_profit=None):
    global cash, portfolio, trade_history, pnl, asset_value
    delay = randint(1, 3)  # Random delay of 1-3 seconds
    await asyncio.sleep(delay)
    slippage = uniform(-0.01, 0.01)  # Random slippage of -1% to 1%
    price *= (1 + slippage)
    if order_type == 'buy':
        quantity = cash / price
        if quantity > 0:
            cash -= quantity * price
            portfolio[symbol] += quantity
            trade_history.append((symbol, order_type, quantity, price, datetime.now()))
            print(f"Bought {quantity} of {symbol} at price {price}")
    elif order_type == 'sell':
        quantity = portfolio[symbol]
        if quantity > 0:
            cash += quantity * price
            pnl += quantity * price - portfolio[symbol] * price
            portfolio[symbol] = 0
            trade_history.append((symbol, order_type, quantity, price, datetime.now()))
            print(f"Sold {quantity} of {symbol} at price {price}")
    asset_value = cash + sum([portfolio[coin] * await get_current_price(coin) for coin in coins])

async def place_orders(symbol, support, resistance, threshold):
    current_price = await get_current_price(symbol)
    if abs(current_price - support) <= threshold:
        await place_order(symbol, 'buy', current_price)
    elif abs(current_price - resistance) <= threshold:
        await place_order(symbol, 'sell', current_price)
    elif current_price > resistance:
        await place_order(symbol, 'buy', current_price)
    elif current_price < support:
        await place_order(symbol, 'sell', current_price)

async def main(symbol, interval='5minute'):
    historical_data = r.crypto.get_crypto_historicals(symbol, interval=interval, span='week')
    closing_prices = [float(x['close_price']) for x in historical_data]
    support = min(closing_prices)
    resistance = max(closing_prices)
    print("Resistance: ", resistance)
    print(f'Current price: {await get_current_price(symbol)}')
    print("Support: ", support)

    await place_orders(symbol, support, resistance, threshold=0.01)

while True:
    print(f'Running at {datetime.now()}')
    for coin in coins:
        asyncio.run(main(coin))
    print(f"Current cash: {cash}")
    print(f"Current portfolio: {portfolio}")
    print(f"Trade history: {trade_history}")
    print(f"Profit and loss: {pnl}")
    print(f"Asset value: {asset_value}")
    time.sleep(60*10)  # Run every ten minutes
