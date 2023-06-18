import asyncio
from datetime import datetime
from colorama import Fore
from tqdm import tqdm
from robin_stocks import robinhood as r
import pandas as pd
import json
from pytz import timezone
from ratelimit import limits, sleep_and_retry
from icecream import ic

BUYING_POWER = [0.00]
PERCENTAGE_IN_PLAY = 0.60
coins_list = [] # the list of coins to trade
minimum_orders_coins = pd.read_csv('data/minimum_orders_coins.csv', index_col='symbol')
def setup():
    global coins_list
    global minimum_orders_coins
    with open('data/CRYPTO_I_OWN.json') as f:
        crypto_i_own = json.load(f)
    BUYING_POWER[0] = float(r.profiles.load_account_profile(info=None)['onbp'])
    with open('data/TOTAL_CRYPTO_DOLLARS.json') as f:
        total_crypto_dollars = json.load(f)
    with open('data/holdings_df.json') as f:
        holdings_df = json.load(f)
    with open('config/credentials.json') as f:
        credentials = json.load(f)
    r.login(username=credentials['username'], password=credentials['password'], expiresIn=86400, by_sms=True)
    coins_list = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']
    return r

@sleep_and_retry
async def get_account():
    account = await asyncio.to_thread(r.profiles.load_account_profile, info=None)
    return account

@sleep_and_retry
async def update_buying_power(buying_power):
    global BUYING_POWER
    while True:
        account_details_df = pd.DataFrame(await asyncio.to_thread(r.profiles.load_account_profile, info=None), index=[0])
        buying_power[0] = float(account_details_df['onbp'])
        if buying_power[0] > 1.00:
            buying_power[0] *= PERCENTAGE_IN_PLAY
        await asyncio.sleep(180)
@sleep_and_retry
async def get_total_crypto_dollars(total_crypto_dollars, crypto_i_own):
    global coins_list
    global BUYING_POWER
    print(Fore.BLUE + 'Updating the portfolio value...' + Fore.RESET)
    for coin in tqdm(crypto_i_own):
        await asyncio.sleep(1.0)
        symbol = coin
        current_price = float(r.crypto.get_crypto_quote(symbol=symbol)['mark_price'])
        number_of_coins = float(crypto_i_own[coin])
        total_value = current_price * number_of_coins
        total_crypto_dollars[0] += float(total_value)
    print(Fore.GREEN + f'\t\tTOTAL_CRYPTO_DOLLARS (c): ${total_crypto_dollars[0]}')

    if is_daytime():
        await asyncio.sleep(170)
    else:
        await asyncio.sleep(300)

def is_daytime():

    current_time = datetime.now(timezone('US/Central'))
    current_hour = current_time.hour
    return current_hour >= 8 and current_hour <= 20
@sleep_and_retry
async def get_crypto_positions_in_account():
    """
    The get_crypto_positions_in_account function returns a dictionary of the crypto positions in your account.
    The keys are the currency codes, and the values are dictionaries containing information about each position.

    :return: A dictionary of all the crypto positions in your account
    :doc-author: Trelent
    """
    global coins_list
    global BUYING_POWER
    crypto_positions = r.crypto.get_crypto_positions(info=None)
    crypto_positions_dict = {}
    for coin in crypto_positions:
        crypto_positions_dict[coin['currency']['code']] = coin
    return crypto_positions_dict

async def brain_module(buying_power, crypto_i_own, coins_list):
    pbar = tqdm(total=len(coins_list))

    while True:
        coin_data = []
        for coin in coins_list:
            if coin in crypto_i_own:
                coin_data.append(f'{coin}: {Fore.CYAN}{round(crypto_i_own[coin], 6)}{Fore.RESET}')
            else:
                coin_data.append(f'{coin}: 0')

        print(Fore.BLUE + 'Crypto Positions: ', coin_data)
        await update_buying_power(buying_power)

        for coin in coins_list:
            current_price = float(r.crypto.get_crypto_quote(symbol=coin)['mark_price'])
            quantity_in_crypto = float(r.crypto.get_crypto_positions()[coin]['quantity'])
            if quantity_in_crypto > minimum_orders_coins.loc[coin]['minimum_order_size'] and coin in crypto_i_own and current_price > (1.05 * float(holdings_df[coin])):
                await sell_crypto(coin, holdings_df[coin])
                print(Fore.YELLOW + f'Selling {coin} - {round(current_price, 2)} > {round(1.05 * holdings_df[coin], 2)}')
                await update_buying_power(buying_power)
            elif quantity_in_crypto > minimum_orders_coins.loc[coin]['minimum_order_size'] and coin not in crypto_i_own and current_price < (float(buying_power[0]) / float(quantity_in_crypto)):
                await buy_crypto(coin, quantity_in_crypto)
                print(Fore.GREEN + f'Buying {coin} - {round(current_price, 2)} < {round(buying_power[0] / quantity_in_crypto, 2)}')
                await update_buying_power(buying_power)

        if is_daytime():
            await asyncio.sleep(170)
        else:
            await asyncio.sleep(300)
        pbar.update(1)

    pbar.close()
@sleep_and_retry
async def buy_crypto(symbol, quantity_in_crypto, buying_power, crypto_i_own):
    """
    The buy_crypto function is used to buy crypto.
        It takes in the following parameters:
            symbol - The ticker symbol of the cryptocurrency you want to buy.
            quantity_in_crypto - How much of that cryptocurrency you want to buy, in terms of that currency (e.g., if I wanted 1 Bitcoin, this would be 1).  This is a float value.
            buying_power - A list containing how much money we have available for buying crypto with (the first element) and how much money we have available for selling crypto with (the second element).  This is a list containing two floats:

    :param symbol: Specify which crypto you want to sell
    :param quantity_in_crypto: Determine how much crypto you want to buy
    :param buying_power: Check if the user has enough buying power to buy the amount of crypto they want
    :param crypto_i_own: Update the json file with the new crypto balance
    :return: The response from the robinhood api
    :doc-author: Trelent
    """

    # global minimum_orders_coins
    if float(buying_power[0]) > float(quantity_in_crypto):
        response = await asyncio.to_thread(r.orders.order_buy_crypto_limit, symbol, float(buying_power[0]) / float(quantity_in_crypto), float(quantity_in_crypto))
        print(response)
        order_details = await asyncio.to_thread(r.orders.order_get_crypto_quote, response['id'])
        if order_details['state'] == 'filled':
            crypto_i_own[symbol] = float(quantity_in_crypto)
            with open('data/CRYPTO_I_OWN.json', 'w') as fp:
                json.dump(crypto_i_own, fp)
@sleep_and_retry
async def sell_crypto(symbol, quantity_in_crypto, crypto_i_own):
    """
    The sell_crypto function is used to sell crypto.
        It takes in a symbol, quantity_in_crypto, and crypto_i_own as arguments.
        The function then uses the robinhood api to place an order for the specified amount of cryptocurrency at market price.
        If the order is filled it removes that cryptocurrency from our list of cryptocurrencies we own.

    :param symbol: Specify which crypto you want to sell
    :param quantity_in_crypto: Specify how much of the crypto you want to sell
    :param crypto_i_own: Update the crypto_i_own
    :return: The order details of the sell order
    :doc-author: Trelent
    """
    ic()
    response = await asyncio.to_thread(r.orders.order_sell_crypto_limit, symbol, float(quantity_in_crypto), float(quantity_in_crypto))
    print(response)
    order_details = await asyncio.to_thread(r.orders.order_get_crypto_quote, response['id'])
    if order_details['state'] == 'filled':
        crypto_i_own.pop(symbol)
        with open('data/CRYPTO_I_OWN.json', 'w') as fp:
            json.dump(crypto_i_own, fp)
@sleep_and_retry
async def get_total_crypto_value_in_dollars(total_crypto_value_in_dollars, crypto_i_own):
    ic()
    while True:
        total_value = 0
        for coin in crypto_i_own:
            total_value += float(r.crypto.get_crypto_quote(symbol=coin)['mark_price']) * float(crypto_i_own[coin])
        total_crypto_value_in_dollars[0] = total_value
        await asyncio.sleep(60)
@sleep_and_retry
async def get_total_crypto_dollars(total_crypto_dollars, crypto_i_own):
    ic()
    while True:
        total_value = 0
        for coin in crypto_i_own:
            total_value += float(r.crypto.get_crypto_quote(symbol=coin)['mark_price']) * float(crypto_i_own[coin])
        total_crypto_dollars[0] = total_value
        await asyncio.sleep(60)

# async function to check buying power every 3 minutes
@sleep_and_retry
async def update_buying_power():
    """
    The update_buying_power function is a coroutine that updates the buying power of the account every 3 minutes.
    It does this by querying Robinhood's API for the current buying power, and then updating it in real time.
    This function is important because it allows us to know how much money we have available to buy crypto with.

    :return: The buying power of the account
    :doc-author: Trelent
    """
    global BUYING_POWER
    ic()
    while True:
        account_details_df = pd.DataFrame(await asyncio.to_thread(r.profiles.load_account_profile, info=None), index=[0])
        BUYING_POWER = float(account_details_df['onbp'])
        # remember to only use the percentage_in_play of the buying power to buy crypto
        if BUYING_POWER > 1.00:
            BUYING_POWER = BUYING_POWER * PERCENTAGE_IN_PLAY
        else:
            BUYING_POWER = BUYING_POWER
        print(Fore.BLUE + f'BUYING_POWER: {BUYING_POWER}')
        await asyncio.sleep(180) # sleep for 3 minutes



def get_minimum_orders_coins():
    """
    The get_minimum_orders_coins function reads in the MINIMUM_ORDERS_COINS.csv file and returns a pandas dataframe
        containing the minimum order sizes for each coin.

    :return: A dataframe with the minimum orders for each coin
    :doc-author: Trelent
    """
    ic()
    minimum_orders_coins = pd.read_csv('data/MINIMUM_ORDERS_COINS.csv', index_col=0)
    return minimum_orders_coins

def is_daytime():
    """
    The is_daytime function returns a boolean value.
    It checks the current time and if it is between 9am and 4pm, then it returns True.
    Otherwise, it returns False.

    :return: True if the current time is between 9am and 4pm, otherwise it returns false
    :doc-author: Trelent
    """
    ic()
    now = datetime.now()
    if now.hour >= 9 and now.hour < 16:
        return True
    else:
        return False

async def main():
    """
    The main function is the main function. It runs all of the other functions in this program.
    It also sets up a few variables that are used throughout the program, such as buying_power and total_crypto_dollars.

    :return: The total crypto value in dollars
    :doc-author: Trelent
    """
    ic()
    global r
    global coins_list
    buying_power = [0.0]
    total_crypto_dollars = [0.0]
    total_crypto_value_in_dollars = [0.0]
    crypto_i_own = {}

    await get_account()
    coins_list = coins_list
    minimum_orders_coins = get_minimum_orders_coins()
    await asyncio.gather(
        brain_module(buying_power, crypto_i_own, coins_list),
        get_total_crypto_dollars(total_crypto_dollars, crypto_i_own),
        get_total_crypto_value_in_dollars(total_crypto_value_in_dollars, crypto_i_own)
    )

if __name__ == "__main__":
    r = setup()
    asyncio.run(main())
