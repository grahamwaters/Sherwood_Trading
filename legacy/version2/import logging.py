import logging
import time
import datetime
import traceback
from tqdm import tqdm
import pandas as pd
import numpy as np

class CryptoTrader:
    def __init__(self, cryptos, username, password):
        self.r = None
        self.logger = None
        self.username = username
        self.password = password
        self.cryptos = cryptos
        self.panic_mode = False
        self.crypto_data = pd.DataFrame()
        self.setup_logging()
        self.login()

    def setup_logging(self):
        self.logger = logging.getLogger('crypto_trader')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('crypto_trader.log')
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    def login(self):
        try:
            self.r.login(self.username, self.password)
        except Exception as e:
            self.logger.error(f'Error during login: {e}')
            traceback.print_exc()

    def get_crypto_data(self):
        for crypto in tqdm(self.cryptos):
            try:
                self.get_individual_crypto_data(crypto)
            except Exception as e:
                self.logger.error(f'Error getting data for {crypto}: {e}')
                traceback.print_exc()

    def get_individual_crypto_data(self, crypto):
        # Get historical data for the given crypto
        historicals = self.r.get_crypto_historicals(crypto, interval='day', span='week')
        
        # Initialize empty list to store the data
        data = []
        
        # Iterate over the historical data
        for history in historicals:
            # Each history entry is a dictionary, so we extract the relevant information
            # For instance, 'begins_at' is the timestamp, 'open_price' is the price at the start of the interval,
            # 'close_price' is the price at the end of the interval, etc.
            # Please modify the fields according to the response structure from the API
            timestamp = history['begins_at']
            open_price = history['open_price']
            close_price = history['close_price']
            high_price = history['high_price']
            low_price = history['low_price']
            
            # Append the data to our list
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'close': close_price,
                'high': high_price,
                'low': low_price,
            })
        
        # Return the data
        return data

    def order_crypto(symbol, quantity_or_price, amount_in='dollars', side='buy'):
        """
        The order_crypto function is used to place a crypto order.
        :param symbol: Specify the coin you want to buy
        :param quantity_or_price: Specify the amount of crypto you want to buy or sell
        :param amount_in: Specify the amount of money you want to spend on buying a coin
        :param side: Specify whether you want to buy or sell the coin
        :return: A dictionary with the following keys:
        :doc-author: Trelent
        """
        if side == 'buy':
            amountIn = 'dollars'
        else:
            amountIn = 'quantity'
        quantity_or_price = str(quantity_or_price)
        try:
            r.orders.order_crypto(
                symbol=str(symbol),
                quantityOrPrice=float(quantity_or_price),
                amountIn=str(amount_in),
                side=str(side),
                timeInForce='gtc',
                jsonify=True
            )
        except Exception as e:
            raise e
    
    def get_crypto_positions(self):
        """
        The get_crypto_positions function is used to get the current crypto positions.
        :return: A dictionary with the following keys:
        :doc-author: Trelent
        """
        try:
            positions = r.crypto.get_crypto_positions()
        except Exception as e:
            raise e
        return positions

    def get_crypto_quote(self, symbol):
        """
        The get_crypto_quote function is used to get the current crypto quote.
        :param symbol: Specify the coin you want to get the quote for
        :return: A dictionary with the following keys:
        :doc-author: Trelent
        """
        try:
            quote = r.crypto.get_crypto_quote(symbol)
        except Exception as e:
            raise e
        return quote

    def get_crypto_quote_from_id(self, id):
        """
        The get_crypto_quote_from_id function is used to get the current crypto quote from the id.
        :param id: Specify the coin id you want to get the quote for
        :return: A dictionary with the following keys:
        :doc-author: Trelent
        """
        try:
            quote = r.crypto.get_crypto_quote_from_id(id)
        except Exception as e:
            raise e
        return quote

    def get_crypto_order_info(self, order_id):
        """
        The get_crypto_order_info function is used to get the current crypto order info.
        :param order_id: Specify the order id you want to get the info for
        :return: A dictionary with the following keys:
        :doc-author: Trelent
        """
        try:
            order_info = r.orders.get_crypto_order_info(order_id)
        except Exception as e:
            raise e
        return order_info

    def get_crypto_orders(self):
    
def main():
    cryptos = ['BTC', 'ETH', 'ADA', 'DOGE', 'MATIC', 'SHIB', 'ETC', 'UNI', 'AAVE', 'LTC', 'LINK', 'COMP', 'USDC', 'SOL', 'AVAX', 'XLM', 'BCH', 'XTZ']
    trader = CryptoTrader(cryptos, 'username', 'password')
    trader.get_crypto_data()

if __name__ == '__main__':
    main()
