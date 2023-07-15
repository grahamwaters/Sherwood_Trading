Instructions for a Refactorization:
Be sure to create asyncronous functions for the different methods of the class structure, allowing them to run in parallel with each other, lowering the number of unnecessary API calls to Robinhood's servers, also, be sure to use the asyncio module to run the functions asynchronously.

Specific Modifications:
1. Functions that are not used in the main.py file should be moved to a separate file called `functions.py` and imported into the main.py file.
2. The classes can be stored in their own files and imported into the main.py file. The classes should be stored in a directory called `classes` and each class should be stored in its own file.
3. The functions that are used in the main.py file should be moved to the main.py file.
4. The functions that are used in the main.py file should be converted to async functions and run asynchronously using the asyncio module.
5. Remember to consider microservices architecture when refactoring the code, and consider how the code can be broken up into smaller pieces that can be run asynchronously and in parallel with each other.
6. I'd like to have a file called `running_config.json` that allows me to change the settings of the bot while it is running, such as the risk tolerance, the take profit percent, the stop loss percent, the coins to trade, etc. I'd like to be able to change these settings without having to restart the bot, just change the settings in the `running_config.json` file and the bot will automatically update the settings and continue running.
7. I'd like the code to include use of alive_progress to show the progress of the bot when it is running, specifically during trading evaluations, and technical indicator calculations, and to show the progress of the bot when it is loading the historical data for the coins. Make the alive_progress bar's aesthetic be updatable from the `running_config.json` file so I can change the aesthetic of the progress bar while the bot is running. I'd like the progress bar to be able to be updated asynchronously while the bot is running.


# Function Modifications:
I need a function:
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
that keeps the log file a certain size (1000 lines) and removes the oldest lines from the log file to maintain a rolling log of 1000 lines.

I also want to add the following to a setup function or something:
"""
stop_loss_percent = 0.05 #^ set the stop loss percent at 5% (of the invested amount)
coins = ['BTC', 'ETH', 'DOGE', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'AVAX', 'XLM', 'BCH', 'XTZ']
"""
Here is the challenge, I don't want the code to rely on selling "ALL" the coins I currently have of say, ETH for example, when I sell. I may have purchased ETH several times and have six "lots" or batches of coins that were purchased at a certain price, and currently are either profitable or losing money given the current_price compared to when I bought the coin.
I need the code to only sell the profitable batches of coins, and not the losing batches of coins when it sells.
This will require a dynamic transaction log dictionary that keeps track of the lots of coins I have purchased and their current profit/loss status.

Required fields:
- lot_uuid (unique identifier for the lot) using the uuid module
- coin (the coin ticker symbol for robinhood) (BTC, ETH, DOGE, etc)
- lot_size (how many coins were purchased in this lot (float))
- lot_cost (how much did it cost in USD to purchase the lot of coins (float))
- lot_profit (how much profit/loss is the lot currently at (float))
- lot_status (is the lot currently open or closed (string))
- lot_open_date (the date/time the lot was opened (datetime))
- lot_close_date (the date/time the lot was closed (datetime))
- lot_open_price (the price the lot was opened at (float)) (use the current_price when the lot is opened) (this is used to calculate the lot_profit)
- lot_close_price (the price the lot was closed at (float)) (use the current_price when the lot is closed) (this is used to calculate the lot_profit) (set to None if the lot is still open)
- lot_sell_trigger (boolean) True or False, and indicates if the lot is currently set to be sold or not based on the take_profit_percent and stop_loss_percent. Should be updated asynchronously when the take_profit_percent or stop_loss_percent is updated.

The transaction log dictionary should look like this:
transaction_log =
  {
      'lots': [
          {
            'lot_uuid': '1234-5678-9012-3456',
            'coin': 'ETH',
            'lot_size': 0.5, #^ 0.5 ETH was purchased
            'lot_cost': 1000.00, #^ how much did it cost in USD to purchase the lot of ETH
            'lot_profit': 0.0, #^ the lot is currently at $0.00 profit
            'lot_status': 'open' #^ the lot is currently open
            'lot_open_date': '2021-01-01 00:00:00' #^ the lot was opened on this date (use datetime.now() to get the current date/time)
            'lot_close_date': '2021-01-01 00:00:00' #^ the lot was closed on this date (use datetime.now() to get the current date/time) (set to None if the lot is still open)
            'lot_open_price': 2000.00 #^ the lot was opened at $2000.00
            'lot_close_price': 0.0 #^ the lot was closed at $0.00 (set to None if the lot is still open)
            'lot_profit_percent': 0.0 #^ the lot is currently at 0.00% profit (set to None if the lot is still open)
            'lot_sell_trigger': False #^ the lot is not currently set to be sold
          }
      ]
  }

save it as a file called `running_transaction_log.json` in the `logs` directory, and asynchronously update it when the take_profit_percent or stop_loss_percent is updated, or when a lot is opened or closed.

I also need a function:
async def transaction_log_updater():
        """
        The transaction_log_updater function is an async function that updates the transaction log with the current profit/loss status of the lots.
        :return: None
        :doc-author: Trelent
        """
        while True:
            #ic()
            # update the transaction log with the current profit/loss status of the lots
            await asyncio.sleep(1200)
that updates the transaction log with the current profit/loss status of the lots.

I also need a function:
async def transaction_log_saver():
        """
        The transaction_log_saver function is an async function that saves the transaction log to a file.
        :return: None
        :doc-author: Trelent
        """
        while True:
            #ic()
            # save the transaction log to a file
            await asyncio.sleep(1200)


# Existing Codebase

<!-- Classes -->
## The Trader Class
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
### functions in the Trader Class
- `login_setup`: The login_setup function logs into Robinhood using the provided username and password.
- `resetter`: The resetter function cancels all open orders and sells all positions. This function is used to reset the bot.
- `calculate_ta_indicators`: The calculate_ta_indicators function calculates different technical indicators and generates trading signals based on these indicators. The indicators are: EMA, MACD, RSI, Williams %R, Stochastic Oscillator, Bollinger Bands, and Parabolic SAR.
        A boolean is generated based on these indicators. If the boolean is True, a buy signal is generated. If the boolean is False, a sell signal is generated. The signals are returned in a DataFrame.
        :param coins: A list of coins to generate signals for
        :return: A DataFrame with the trading signals for each coin
- `trading_function`: The trading_function function executes actions based on the trading signals. If the signal is a buy signal, the coin is bought. If the signal is a sell signal, the coin is sold, if it is owned.
        :param signals: A DataFrame with the trading signals for each coin
        :return: None
- `get_total_crypto_dollars`: The get_total_crypto_dollars function returns the total amount of crypto dollars in the Robinhood account (in USD).
        :return: The total amount of crypto dollars in the Robinhood account (in USD)
- `update_buying_power`: The update_buying_power function updates the buying power of the Robinhood account (in USD).
        :return: None
- `check_stop_loss_prices`: The check_stop_loss_prices function checks if the current price of a coin is below the stop loss price. If it is, the coin is sold.
        :param coins: A list of coins to check the stop loss price for
        :return: None
- `main`: The main function is the "main" or primary function. It will do the following:
    1) Check if there are any open orders that need to be cancelled
    2) Check if there are any positions that need to be sold (if we're in a sell state)
    3) If we're in a buy state, check for new stocks to buy based on our criteria
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :return: None

## The Looper Class
- `run_async_functions`: the main function that runs all of the other functions. The run_async_functions function is the main function that runs all of the other functions.
        It will run them simultaneously, and it will also keep track of how many times it has looped.
        The loop_count variable is used to determine when to update buying power, which happens every 10 loops.

        :param loop_count: Keep track of how many times the loop has run
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :return: A coroutine object

- `main_looper`: the main looper function that runs the bot
        The main_looper function is the main loop function. It will run every hour and do the following:
            1) Check if there are any open orders that need to be cancelled
            2) Check if there are any positions that need to be sold (if we're in a sell state)
            3) If we're in a buy state, check for new stocks to buy based on our criteria
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
### functions in the looper class
- `__init__`: initializes the looper class
- async `main_looper`: the main looper function that runs the bot
  -  """
        The main_looper function is the main loop function. It will run every hour and do the following:
            1) Check if there are any open orders that need to be cancelled
            2) Check if there are any positions that need to be sold (if we're in a sell state)
            3) If we're in a buy state, check for new stocks to buy based on our criteria
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :doc-author: Trelent
        """
- async `run_async_functions`: runs the async functions in the looper class
  - The run_async_functions function is the main function that runs all of the other functions.
        It will run them simultaneously, and it will also keep track of how many times it has looped.
        The loop_count variable is used to determine when to update buying power, which happens every 10 loops.

        :param loop_count: Keep track of how many times the loop has run
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :return: A coroutine object


## The Utility Class
### functions in the utility class
- `__init__`: initializes the utility class
- async `log_file_size_checker`: checks the size of the log file and deletes it if it is too large
  - """
        The log_file_size_checker function is an async function that checks the size of the log file and removes lines from the start of the file to maintain a rolling log of 1000 lines.
        :return: None
        :doc-author: Trelent
        """
- `get_last_100_days`: The get_last_100_days function gets the last 100 days of a particular coin's data.
        :param coin: The coin to get data for
        :return: A DataFrame with the last 100 days of coin data
- `is_daytime`: The is_daytime function checks if it is daytime or nighttime.
        The is_daytime function checks if the current time is between 8 AM and 8 PM.
        :return: True if it's between 8 AM and 8 PM, False otherwise

Instructions for GPT-4
======================
Using what you have saved in MemoryBank (functions and their docstrings), your task is to write a python script that incorporates the requirements mentioned above, and the functionality of the functions in the MemoryBank. You can use the functions in the MemoryBank as a starting point, but you will need to modify them to fit the requirements of the project. If you need clarification at any point, please ask for help. Also, if you stop before you are done use the /auto_continue function to preserve your progress without having to wait for the user to type "continue". You can use the /auto_continue function at any time to save your progress, and don't forget that you can use the /write_custom_function: if needed, to create your own LLM internal GPT functions, displaying it in a code block and add it to your abilities. Include inputs, methods, and outputs specs.
Tell me you understand your instructions, then give me an outline of your plan for this project.