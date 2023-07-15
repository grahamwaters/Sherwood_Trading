import logging
import pandas as pd
import pandas_ta as ta
import robin_stocks as rstocks
from robin_stocks import robinhood as r
from datetime import datetime
from pytz import timezone
import asyncio
from colorama import Fore, Back, Style
import json
import os
from classes import Trader, Looper, Utility
from dotenv import load_dotenv # this is for loading the .env file

# now asyncronously begin the program
async def main():
    # Create a Transaction Log or Load from File
    if not os.path.exists('config/transaction_log.json'):
        with open('config/transaction_log.json', 'w', encoding='utf-8') as f:
            json.dump({}, f)
    else:
        with open('config/transaction_log.json', 'r') as f:
            transaction_log = json.load(f)
    print(Fore.GREEN + 'Transaction Log Loaded' + Fore.RESET)

    # Create a Running Config or Load from File
    if not os.path.exists('config/running_config.json'):
        with open('config/running_config.json', 'w') as f:
            json.dump({}, f)
    else:
        with open('config/running_config.json', 'r') as f:
            running_config = json.load(f)
    print(Fore.GREEN + 'Running Config Loaded' + Fore.RESET)

    # Update the transaction log
    os.environ['transaction_log'] = json.dumps(transaction_log)

    # Update the running config
    os.environ['running_config'] = json.dumps(running_config)

    # use the running config to set the username
    with open('config/credentials.json', 'r') as f:
        credentials = json.load(f)
    username = credentials['username']
    password = credentials['password']

    # Create a Trader Object
    trader = Trader(
        username=username,
        password=password
    )

    # Create a Looper Object
    with open('config/running_config.json', 'r') as f:
        running_config = json.load(f)
    trader.set_running_config(running_config)

    # Create a Looper Object
    looper = Looper()
    # use the running config to set the username
    looper.set_username(running_config['username'])
    looper.set_password(running_config['password'])

    # Create a Robinhood Object
    robinhood = r

    # Login to Robinhood
    robinhood.login(username=os.getenv('RH_USERNAME'), password=os.getenv('RH_PASSWORD'))

    # Get the Account Information
    account = robinhood.load_account_profile()
    print(Fore.GREEN + f'Account Profile Loaded: {account}' + Fore.RESET)

    # Get the Account Balance
    account_balance = float(account['buying_power'])

    # Get the Account Buying Power
    buying_power = float(account['buying_power'])

    # Get the Account Max Coins
    max_coins = int(buying_power / 5)


if __name__ == '__main__':
    asyncio.run(main())
