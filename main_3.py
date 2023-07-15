import asyncio
import json
import os
from colorama import Fore
from classes import Trader, Looper, Utility

def first_time_init():
    # Create a Transaction Log or Load from File
    transaction_log = {}
    if os.path.exists('config/transaction_log.json'):
        with open('config/transaction_log.json', 'r', encoding='utf-8') as f:
            try:
                transaction_log = json.load(f)
            except json.JSONDecodeError as e:
                print(Fore.RED + f'Error loading transaction log: {e}' + Fore.RESET)
    else:
        with open('config/transaction_log.json', 'w', encoding='utf-8') as f:
            json.dump(transaction_log, f)
    print(Fore.GREEN + 'Transaction Log Loaded' + Fore.RESET)

    # Create a Running Config or Load from File
    running_config = {}
    if os.path.exists('config/running_config.json'):
        with open('config/running_config.json', 'r') as f:
            try:
                running_config = json.load(f)
            except json.JSONDecodeError as e:
                print(Fore.RED + f'Error loading running config: {e}' + Fore.RESET)
    else:
        with open('config/running_config.json', 'w') as f:
            json.dump(running_config, f)
    print(Fore.GREEN + 'Running Config Loaded' + Fore.RESET)

    return transaction_log, running_config


async def initialize_and_update():
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


async def main():
    """
    The main function is the entry point for the program.
    It initializes all of the classes and runs their functions asynchronously.

    :return: A future
    :doc-author: Trelent
    """
    # Load credentials
    with open('config/credentials.json', 'r') as f:
        credentials = json.load(f)
    username = credentials['username']
    password = credentials['password']

    # Initialize classes and functions
    trader = Trader(username, password)
    looper = Looper(trader)
    utility = Utility()

    # Load configurations for the session
    with open('config/running_config.json', 'r') as f:
        config = json.load(f)
        coins = config['available_coins']
        buying_power = config['buying_power']
        # max_coins = config['max_coins']
        # min_coins = config['min_coins']
        # max_price = config['max_price']
        # min_price = config['min_price']
        # max_volume = config['max_volume']
        # min_volume = config['min_volume']
        # max_rsi = config['max_rsi']
        # min_rsi = config['min_rsi']
        # max_macd = config['max_macd']
        # min_macd = config['min_macd']
        # max_stoch = config['max_stoch']
        # min_stoch = config['min_stoch']
        # max_williams = config['max_williams']
        # min_williams = config['min_williams']
        # max_mfi = config['max_mfi']
        # min_mfi = config['min_mfi']
        # max_ao = config['max_ao']
        # min_ao = config['min_ao']
        # max_adosc = config['max_adosc']
        # min_adosc = config['min_adosc']
        # max_cci = config['max_cci']
        # min_cci = config['min_cci']
        # max_bbands = config['max_bbands']
        # min_bbands = config['min_bbands']
        # max_adx = config['max_adx']
        # min_adx = config['min_adx']
        # max_aroon = config['max_aroon']
        # min_aroon = config['min_aroon']
    # Set config values
    trader.set_config_values(config)

    # Run functions
    signals_df = await trader.calculate_ta_indicators(coins)

    await asyncio.gather(
        trader.login_setup(),
        trader.resetter(),
        trader.trading_function(signals_df),
        trader.update_buying_power()
    )

# Run main function
if __name__ == '__main__':
    transaction_log, running_config = first_time_init()
    os.environ['transaction_log'] = json.dumps(transaction_log)
    os.environ['running_config'] = json.dumps(running_config)

    asyncio.run(initialize_and_update())

    asyncio.run(main())
