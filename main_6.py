import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
import pandas as pd
import pandas_ta as ta
import robin_stocks as rstocks
from colorama import Back, Fore, Style
from dotenv import load_dotenv
from pytz import timezone
from ratelimit import limits, sleep_and_retry
from robin_stocks import robinhood as r
from tqdm import tqdm
"""
{
    "available_coins": [
        "BTC",
        "ETH",
        "DOGE",
        "SHIB",
        "ETC",
        "UNI",
        "AAVE",
        "LTC",
        "LINK",
        "COMP",
        "USDC",
        "AVAX",
        "XLM",
        "BCH",
        "XTZ"
    ],
    "stop_loss_percent": 0.05,
    "take_profit_percentage": 0.15,
    "buying_power": 101.25,
    "reset_mode": false,
    "verbose_mode": true,
    "trading_mode": "real",
    "trading_fee": 0.0,
    "play_with_pct": 0.4,
    "coin_momentums": {
        "BTC": 0,
        "ETH": 0
    }
}
"""
"""
{
    "BTC": {
        "orders":
        [
            {
                "order_id": "1",
                "order_type": "buy",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }, {
                "order_id": "2",
                "order_type": "sell",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }
        ]
    },
    "ETH": {
        "orders":
        [
            {
                "order_id": "3",
                "order_type": "buy",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }, {
                "order_id": "4",
                "order_type": "sell",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }
        ]
    },
    "DOGE": {
        "orders":
        [
            {
                "order_id": "42",
                "order_type": "buy",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }, {
                "order_id": "69",
                "order_type": "sell",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }
        ]
    },
    "ETC": {
        "orders":
        [
            {
                "order_id": "63",
                "order_type": "buy",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }, {
                "order_id": "74",
                "order_type": "sell",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }
        ]
    },
    "SHIB": {
        "orders":
        [
            {
                "order_id": "12",
                "order_type": "buy",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }, {
                "order_id": "332",
                "order_type": "sell",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }
        ]
    },
    "UNI": {
        "orders":
        [
            {
                "order_id": "14",
                "order_type": "buy",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }, {
                "order_id": "5",
                "order_type": "sell",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }
        ]
    },
    "XRP": {
        "orders":
        [
            {
                "order_id": "781",
                "order_type": "buy",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }, {
                "order_id": "123",
                "order_type": "sell",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "closed"
            }
        ]
    },
    "AAVE": {
        "orders":
        [
            {
                "order_id": "1",
                "order_type": "buy",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }, {
                "order_id": "2",
                "order_type": "sell",
                "order_price": 100000,
                "order_amount": 0.1,
                "order_status": "open"
            }
        ]
    }
}
"""
class Trader:
    """
    The Trader class provides functions for logging into Robinhood, resetting orders, generating trading signals, executing actions based on these signals, updating the buying power, and checking stop loss prices.
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
        self.logger = logging.getLogger("trader")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.login_setup()
        with open("config/running_config.json", "r") as f:
            config = json.load(f)
            try:
                if (
                    config["available_coins"][0] == ""
                    or len(config["available_coins"][0]) == 0
                    or config["available_coins"][0] == " "
                    or config["available_coins"][0] == "[]"
                    or config["available_coins"][0] == "{}"
                ):
                    self.available_coins = 0
                else:
                    self.available_coins = float(config["available_coins"][0])
            except Exception as e:
                self.available_coins = 0
                logging.error(f"Unable to get available coins... {e}")
            self.stop_loss_percent = float(config["stop_loss_percent"])
            self.buying_power = float(config["buying_power"])
            self.take_profit_percentage = float(config["take_profit_percentage"])
            self.percent_to_use = float(
                config["play_with_pct"]
            )
    @sleep_and_retry
    def login_setup(self):
        """
        The login_setup function logs into Robinhood using the provided username and password.
        :doc-author: Trelent
        """
        try:
            r.login(self.username, self.password)
            self.logger.info("Logged in to Robinhood successfully.")
        except Exception as e:
            self.logger.error(f"Unable to login to Robinhood... {e}")
    @sleep_and_retry
    def resetter(self):
        """
        The resetter function cancels all open orders and sells all positions.
        :doc-author: Trelent
        """
        with open("config/running_config.json", "r") as f:
            if json.load(f)["reset_mode"] == "false":
                return
        try:
            open_orders = r.get_all_open_crypto_orders()
            print(Fore.YELLOW + "Canceling all open orders..." + Style.RESET_ALL)
            for order in tqdm(open_orders):
                r.cancel_crypto_order(order["id"])
            print(Fore.GREEN + "All open orders cancelled.")
            self.logger.info("All open orders cancelled." + Style.RESET_ALL)
            crypto_positions = r.get_crypto_positions()
            for position in crypto_positions:
                r.order_sell_crypto_limit(
                    position["currency"]["code"],
                    position["quantity"],
                    position["cost_bases"][0]["direct_cost_basis"],
                )
            self.logger.info("All positions sold.")
        except Exception as e:
            self.logger.error(f"Unable to reset orders and positions... {e}")
    def check_stop_loss_prices(self, coins):
        """
        The check_stop_loss_prices function checks if the current price of a coin is below the stop loss price. If it is, the coin is sold.
        :param coins: A list of coins to check the stop loss price for
        :return: None
        :doc-author: Trelent
        """
        try:
            crypto_positions = r.get_crypto_positions()
            for position in tqdm(crypto_positions):
                if position["currency"]["code"] in coins:
                    current_price = float(
                        r.get_crypto_quote(position["currency"]["code"], "mark_price")[
                            "mark_price"
                        ]
                    )
                    stop_loss_price = float(
                        position["cost_bases"][0]["direct_cost_basis"]
                    ) * (1 - self.stop_loss_percent)
                    if current_price <= stop_loss_price:
                        r.order_sell_crypto_limit(
                            position["currency"]["code"],
                            position["quantity"],
                            position["cost_bases"][0]["direct_cost_basis"],
                        )
                        self.logger.info(
                            f'Stop loss triggered for {position["currency"]["code"]}.'
                        )
        except Exception as e:
            self.logger.error(f"Unable to check stop loss prices... {e}")
    def update_buying_power(self):
        """
        The update_buying_power function updates the buying power from the Robinhood account and saves it to the `running_config.json` file in the `config` directory.
        :return: None
        :doc-author: Trelent
        """
        try:
            self.buying_power = float(r.load_account_profile()["crypto_buying_power"])
            with open("config/running_config.json", "r") as f:
                config = json.load(f)
                config["buying_power"] = self.buying_power
            with open("config/running_config.json", "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Unable to update buying power... {e}")
    def update_available_coins(self):
        """
        The update_available_coins function updates the available coins from the Robinhood account and saves it to the `running_config.json` file in the `config` directory.
        :return: None
        :doc-author: Trelent
        """
        try:
            self.available_coins = float(
                r.load_account_profile()["crypto_quantity_available"]
            )
            with open("config/running_config.json", "r") as f:
                config = json.load(f)
                config["available_coins"] = self.available_coins
            with open("config/running_config.json", "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Unable to update available coins... {e}")
    def update_take_profit_percentage(self):
        """
        The update_take_profit_percentage function updates the take profit percentage from the Robinhood account and saves it to the `running_config.json` file in the `config` directory.
        :return: None
        :doc-author: Trelent
        """
        try:
            self.take_profit_percentage = float(
                r.load_account_profile()["crypto_max_liquidation_price_multiplier"]
            )
            with open("config/running_config.json", "r") as f:
                config = json.load(f)
                config["take_profit_percentage"] = self.take_profit_percentage
            with open("config/running_config.json", "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Unable to update take profit percentage... {e}")
    def update_stop_loss_percent(self):
        """
        The update_stop_loss_percent function updates the stop loss percentage from the Robinhood account and saves it to the `running_config.json` file in the `config` directory.
        :return: None
        :doc-author: Trelent
        """
        try:
            self.stop_loss_percent = float(
                r.load_account_profile()["crypto_max_liquidation_price_multiplier"]
            )
            with open("config/running_config.json", "r") as f:
                config = json.load(f)
                config["stop_loss_percent"] = self.stop_loss_percent
            with open("config/running_config.json", "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Unable to update stop loss percentage... {e}")
    def calculate_ta_indicators(self, coins):
        """
        The calculate_ta_indicators function calculates different technical indicators and generates trading signals based on these indicators. The indicators are: EMA, MACD, RSI, Williams %R, Stochastic Oscillator, Bollinger Bands, and Parabolic SAR.
        A boolean is generated based on these indicators. If the boolean is True, a buy signal is generated. If the boolean is False, a sell signal is generated. The signals are returned in a DataFrame.
        :param coins: A list of coins to generate signals for
        :return: A DataFrame with the trading signals for each coin
        """
        try:
            utility = Utility()
            signals_df = pd.DataFrame()
            coin_lot_signals = (
                {}
            )
            if os.path.exists("config/transaction_log.json"):
                with open("config/transaction_log.json", "r") as f:
                    transaction_log = json.load(f)
                    for coin in transaction_log:
                        coin_lot_signals[coin] = transaction_log[coin]
            for coin in coins:
                df = utility.get_last_100_days(coin)
                current_price = float(df.close.iloc[-1])
                df["sma"] = df.close.rolling(window=50).mean()
                df["ema"] = df.close.ewm(span=50, adjust=False).mean()
                df["macd_line"], df["signal_line"], df["macd_hist"] = ta.macd(df.close)
                df["rsi"] = ta.rsi(df.close)
                df[
                    "current_price"
                ] = current_price
                df["purchase_price"] = coin_lot_signals[coin][
                    0
                ]
                df["stop_loss_price"] = df["purchase_price"] * (
                    1 - self.stop_loss_percent
                )
                df["take_profit_price"] = df["purchase_price"] * (
                    1 + self.take_profit_percentage
                )
                signals_df = signals_df.append(df)
            return signals_df
        except Exception as e:
            self.logger.error(f"Unable to generate trading signals... {e}")
            return pd.DataFrame()
    def set_config_values(self, config):
        file = "config/running_config.json"
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                try:
                    existing_config = json.load(f)
                    existing_config.update(config)
                    with open(file, "w", encoding="utf-8") as f:
                        json.dump(existing_config, f, indent=4)
                except Exception as e:
                    self.logger.error(f"Unable to set config values... {e}")
        else:
            with open(file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
        logging.info(
            "Config values have been set. The bot will now restart with the new values."
        )
    def get_config_values(self):
        with open("config/running_config.json", "r", encoding="utf-8") as f:
            try:
                config = json.load(f)
                return config
            except Exception as e:
                self.logger.error(f"Unable to get config values... {e}")
                return {}
    @sleep_and_retry
    def trading_function(self, signals_df):
        """
        The trading_function function takes the trading signals generated by calculate_ta_indicators() and places trades accordingly.
        :param signals_df: A DataFrame with the trading signals for each coin
        :doc-author: Trelent
        """
        try:
            crypto_positions = r.get_crypto_positions()
            time.sleep(random.randint(1, 3))
            for _, row in signals_df.iterrows():
                coin = row["coin"]
                close_price = row["close"]
                if row["buy_signal"]:
                    Quant = 0
                    with open("config/running_config.json", "r", encoding="utf-8") as f:
                        try:
                            config = json.load(f)
                            coin_momentums = config.get("coin_momentums", {})
                            momentum = coin_momentums.get(coin, 0)
                            Quant = float(
                                self.buying_power
                                * self.percent_to_use
                                / close_price
                                * (1 / momentum)
                            )
                        except Exception as e:
                            print(
                                Fore.RED
                                + f"Unable to load running_config.json... {e}"
                                + Style.RESET_ALL
                            )
                    if Quant == 0:
                        Quant = self.buying_power / close_price * self.percent_to_use
                    block_text = f"""
                    {coin} bought at {close_price} because:
                    - MACD Line: {row['macd_line']}
                    - Signal Line: {row['signal_line']}
                    - RSI: {row['rsi']}
                    - Williams %R: {row['williams']}
                    - Stochastic K: {row['stochastic_k']}
                    - Stochastic D: {row['stochastic_d']}
                    """
                    print(Fore.GREEN + block_text + Style.RESET_ALL)
                    buying_power = self.update_buying_power()
                    if buying_power > 0:
                        r.order_buy_crypto_limit(
                            symbol=coin,
                            quantity=Quant,
                            limitPrice=close_price,
                            timeInForce="gtc",
                        )
                        self.logger.info(f"Bought {coin} at {close_price}.")
                        print(
                            Fore.GREEN
                            + f"Bought {coin} at {close_price}."
                            + Style.RESET_ALL
                        )
                        with open("config/transaction_log.json", "r") as f:
                            data = json.load(f)
                            data["transactions"].append(
                                {
                                    "lot_coin": coin,
                                    "lot_purchase_price": close_price,
                                    "lot_quantity": buying_power / close_price,
                                    "timestamp": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                }
                            )
                if row["sell_signal"] or row["stop_loss"] or row["take_profit"]:
                    Quant = 0
                    for position in crypto_positions:
                        if (
                            position["currency"]["code"] == coin
                            and float(position["quantity"]) > 0
                        ):
                            with open(
                                "config/transaction_log.json", "r", encoding="utf-8"
                            ) as f:
                                data = json.load(f)
                                for transaction in data["transactions"]:
                                    if (
                                        transaction["lot_coin"] == coin
                                        and float(transaction["lot_purchase_price"])
                                        < close_price
                                    ):
                                        Quant += float(transaction["lot_quantity"])
                                if len(data["transactions"]) == 0:
                                    if float(position["quantity"]) > 0:
                                        Quant += float(position["quantity"])
                    if Quant > 0:
                        r.order_sell_crypto_limit(
                            symbol=coin,
                            quantity=Quant,
                            limitPrice=close_price,
                            timeInForce="gtc",
                        )
                        self.logger.info(f"Sold {coin} at {close_price}.")
                        print(
                            Fore.RED
                            + f"Sold {coin} at {close_price}."
                            + Style.RESET_ALL
                        )
                        with open(
                            "config/transaction_log.json", "r", encoding="utf-8"
                        ) as f:
                            data = json.load(f)
                            data["transactions"].append(
                                {
                                    "lot_coin": coin,
                                    "lot_purchase_price": close_price,
                                    "lot_quantity": Quant,
                                    "timestamp": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                }
                            )
        except Exception as e:
            self.logger.error(f"Unable to execute trades... {e}")
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
                total_crypto_dollars += float(position["quantity"]) * float(
                    r.crypto.get_crypto_quote(position["currency"]["code"])[
                        "mark_price"
                    ]
                )
            return total_crypto_dollars
        except Exception as e:
            self.logger.error(f"Unable to get total value of crypto... {e}")
            return 0
    def main(self, coins, stop_loss_prices):
        """
        The main function is the main function. It will do the following:
            1) Check if there are any open orders that need to be cancelled
            2) Check if there are any positions that need to be sold (if we're in a sell state)
            3) If we're in a buy state, check for new stocks to buy based on our criteria
        :param coins: A list of coins to check
        :param stop_loss_prices: A dictionary with the stop loss price for each coin
        :return: The main function
        :doc-author: Trelent
        """
        try:
            utility = Utility()
            if utility.is_daytime():
                self.resetter()
                signals_df = self.calculate_ta_indicators(coins)
                self.trading_function(signals_df)
                self.check_stop_loss_prices(coins, stop_loss_prices)
            else:
                self.logger.info("It is not daytime. The main function will not run.")
        except Exception as e:
            self.logger.error(f"Unable to run main function... {e}")
class Utility:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    async def log_file_size_checker(self):
        """
        The log_file_size_checker function is an async function that checks the size of the log file and removes lines from the start of the file to maintain a rolling log of 1000 lines.
        :return: None
        :doc-author: Trelent
        """
        while True:
            with open("logs/robinhood.log", "r") as f:
                lines = f.readlines()
                if len(lines) > 1000:
                    num_lines_to_remove = len(lines) - 1000
                    with open("logs/robinhood.log", "w") as f:
                        f.writelines(lines[num_lines_to_remove:])
            print("Log file size checked and reduced to 1000 lines.")
            await asyncio.sleep(1200)
    def get_last_100_days(self, coin):
        """
        The get_last_100_days function gets the last 100 days of a particular coin's data.
        :param coin: The coin to get data for
        :return: A DataFrame with the last 100 days of coin data
        :doc-author: Trelent
        """
        try:
            df = pd.DataFrame(
                r.crypto.get_crypto_historicals(
                    coin, interval="hour", span="3month", bounds="24_7"
                )
            )
            df = df.set_index("begins_at")
            df.index = pd.to_datetime(df.index)
            df = df.loc[:, ["close_price", "open_price", "high_price", "low_price"]]
            df = df.rename(
                columns={
                    "close_price": "close",
                    "open_price": "open",
                    "high_price": "high",
                    "low_price": "low",
                }
            )
            df = df.apply(pd.to_numeric)
            return df
        except Exception as e:
            print(f"Unable to get data for {coin}... {e}")
            return pd.DataFrame()
    def is_daytime(self):
        """
        The is_daytime function checks if the current time is between 8 AM and 8 PM.
        :return: True if it's between 8 AM and 8 PM, False otherwise
        :doc-author: Trelent
        """
        current_time = datetime.now(timezone("US/Central"))
        current_hour = current_time.hour
        if current_hour >= 8 and current_hour <= 20:
            return True
        else:
            return False
    async def check_config(self):
        with open("config/running_config.json", "r") as f:
            config = json.load(f)
            return (
                config["available_coins"],
                config["stop_loss_percent"],
                config["buying_power"],
                config["percent_to_use"],
            )
class Looper:
    def __init__(self, trader: Trader):
        """
        The Looper class provides functions for running asynchronous operations.
        :param trader: An instance of the Trader class
        :doc-author: Trelent
        """
        self.trader = trader
        self.logger = logging.getLogger("looper")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
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
                self.trader.update_buying_power()
            self.trader.main(coins, stop_loss_prices)
            self.trader.log_file_size_checker()
        except Exception as e:
            self.logger.error(f"Unable to run async functions... {e}")
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
                await self.run_async_functions(loop_count, coins, stop_loss_prices)
                loop_count += 1
                await asyncio.sleep(3600)
            except Exception as e:
                self.logger.error(f"Error in main loop... {e}")
    def set_username(self, username):
        """
        The set_username function sets the username for the Trader class.
        :param username: The username to set
        :doc-author: Trelent
        """
        self.trader.username = username
    def set_password(self, password):
        """
        The set_password function sets the password for the Trader class.
        :param password: The password to set
        :doc-author: Trelent
        """
        self.trader.password = password
    def set_buying_power(self, buying_power):
        """
        The set_buying_power function sets the buying power for the Trader class.
        :param buying_power: The buying power to set
        :doc-author: Trelent
        """
        self.trader.buying_power = buying_power
    def set_stop_loss_percent(self, stop_loss_percent):
        """
        The set_stop_loss_percent function sets the stop loss percentage for the Trader class.
        :param stop_loss_percent: The stop loss percentage to set
        :doc-author: Trelent
        """
        self.trader.stop_loss_percent = stop_loss_percent
async def main():
    if not os.path.exists("config/transaction_log.json"):
        with open("config/transaction_log.json", "w", encoding="utf-8") as f:
            json.dump({}, f)
    else:
        with open("config/transaction_log.json", "r") as f:
            transaction_log = json.load(f)
    print(Fore.GREEN + "Transaction Log Loaded" + Fore.RESET)
    if not os.path.exists("config/running_config.json"):
        with open("config/running_config.json", "w") as f:
            json.dump({}, f)
    else:
        with open("config/running_config.json", "r") as f:
            running_config = json.load(f)
    print(Fore.GREEN + "Running Config Loaded" + Fore.RESET)
    os.environ["transaction_log"] = json.dumps(transaction_log)
    os.environ["running_config"] = json.dumps(running_config)
    with open("config/credentials.json", "r") as f:
        credentials = json.load(f)
    username = credentials["username"]
    password = credentials["password"]
    trader = Trader(username=username, password=password)
    with open("config/running_config.json", "r") as f:
        running_config = json.load(f)
    trader.set_running_config(running_config)
    looper = Looper()
    looper.set_username(running_config["username"])
    looper.set_password(running_config["password"])
    robinhood = r
    robinhood.login(
        username=os.getenv("RH_USERNAME"), password=os.getenv("RH_PASSWORD")
    )
    account = robinhood.load_account_profile()
    print(Fore.GREEN + f"Account Profile Loaded: {account}" + Fore.RESET)
    account_balance = float(account["buying_power"])
    buying_power = float(account["buying_power"])
    max_coins = int(buying_power / 5)
if __name__ == "__main__":
    with open("config/running_config.json", "r") as f:
        stop_loss_prices = json.load(f)["stop_loss_prices"]
    with open("config/running_config.json", "r") as f:
        stop_loss_percent = json.load(f)["stop_loss_percent"]
    coins = [
        "BTC",
        "ETH",
        "DOGE",
        "SHIB",
        "ETC",
        "UNI",
        "AAVE",
        "LTC",
        "LINK",
        "COMP",
        "USDC",
        "AVAX",
        "XLM",
        "BCH",
        "XTZ",
    ]
    with open("config/credentials.json") as f:
        credentials = json.load(f)
    username = credentials["username"]
    password = credentials["password"]
    trader_object = Trader(username="username", password="password")
    stop_loss_prices = {
        coin: float(r.crypto.get_crypto_quote(coin)["mark_price"])
        - (float(r.crypto.get_crypto_quote(coin)["mark_price"]) * stop_loss_percent)
        for coin in coins
    }
    print(f"Stop loss prices: {stop_loss_prices}")
    trader = Trader(
        "username", "password"
    )
    looper = Looper(
        trader
    )
    asyncio.run(looper.main_looper(coins, stop_loss_prices))
