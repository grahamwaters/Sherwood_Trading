import json
import random
import time
import requests
import spacy
import nltk
from bs4 import BeautifulSoup
from colorama import Back, Fore, Style
from fuzzywuzzy import fuzz
from icecream import ic
from newspaper import Article, Config, ArticleException
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ratelimit import limits, sleep_and_retry
from robin_stocks import robinhood as r
from tqdm import tqdm
import pandas as pd
print(f'Beginning setup...')
import ray
# run this cell to start the ray cluster
# ignore reinitialization errors
ray.shutdown()
ray.init()

nltk.download('vader_lexicon')

# Load secrets
with open('secrets.json') as f:
    secrets = json.load(f)

# Robinhood login
username = secrets['username']
password = secrets['password']
login = r.login(username, password)

# Initialize NLP tools
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

cryptocurrencies = [
    'BTC',
    'ETH',
    'DOGE',
    'LTC',
    'ETC',
    'BCH',
    'BSV',
    'XRP',
    'ADA',
    'DOT',
    'UNI',
    'LINK',
    'XLM'
]

influence_dict = {
    "MSFT": 0.0
    }




# Define utility functions

get_company_names = lambda text: list(set([ent.text for ent in nlp(text).ents if ent.label_ == 'ORG' or ent.text in cryptocurrencies]))
calculate_sentiment = lambda text: sia.polarity_scores(text)['compound']
levenshtein_similarity_check = lambda new_article, articles_list: any(fuzz.ratio(new_article, article) > 80 for article in articles_list)
get_crypto_names = lambda article_text: [crypto for crypto in cryptocurrencies if crypto in article_text]
calculate_sentiment_of_sentence_containing_crypto = lambda article_text: sum(sia.polarity_scores(sentence)["compound"] for sentence in article_text.split('. ') for crypto in cryptocurrencies if crypto in sentence) / len([sentence for sentence in article_text.split('. ') for crypto in cryptocurrencies if crypto in sentence]) if [sentence for sentence in article_text.split('. ') for crypto in cryptocurrencies if crypto in sentence] else 0

@ray.remote
def check_for_trade_signals(sentiment_scores):
    trades = []
    for company, sentiment_score in sentiment_scores.items():
        if sentiment_score > 0.05:  # This is an example threshold for buying
            trades.append((company, 'BUY'))
        elif sentiment_score < -0.05:  # This is an example threshold for selling
            trades.append((company, 'SELL'))
    return trades

# @ray.remote
# def update_influence(company, influence_dict, sentence, sentiment_score):
#     # cast sentiment_score to float
#     sentiment_score = float(sentiment_score)
#     if company in influence_dict:
#         # update the influence_dict with the new sentence
#         # if it's a positive sentiment, add the sentence to the positive list
#         # if it's a negative sentiment, add the sentence to the negative list
#         if sentiment_score > 0:
#             company = str(company).upper()
#             sentence = str(sentence)
#             influence_dict[company]['positive_sentences'].append(sentence)
#         elif sentiment_score < 0:
#             influence_dict[company]['negative_sentences'].append(sentence)
#     else:
#         influence_dict[company] = {
#             "positive_sentences": [],
#             "negative_sentences": [],
#             "overall_sentiment": 0.0,
#             "influence": 0.0
#         }
#         # update the influence_dict with the new sentence
#         # if it's a positive sentiment, add the sentence to the positive list
#         # if it's a negative sentiment, add the sentence to the negative list
#         if sentiment_score > 0:
#             company = str(company).upper()
#             sentence = str(sentence)
#             influence_dict[company]['positive_sentences'].append(sentence)
#         elif sentiment_score < 0:
#             influence_dict[company]['negative_sentences'].append(sentence)
#         else:
#             pass
#     return influence_dict


@ray.remote
def get_articles(url):
    # use newspaper3k to get articles
    paper = newspaper.build(url, memoize_articles=False)
    articles = []
    for article in tqdm(paper.articles):
        # update the tqdm message in blue with the article title
        try:
            tqdm.write(Fore.BLUE + str(article.title))
        except:
            pass
        try:
            article.download()
            article.parse()
            articles.append(article.text)
        except:
            continue
    return articles

@ray.remote
def get_sentiment_scores(articles):
    sentiment_scores = {}
    for article in articles:
        sentiment_scores[article] = calculate_sentiment_of_sentence_containing_crypto(article)
    return sentiment_scores

def getTicker(company_name):
    """
    The getTicker function takes in a company name and returns the ticker symbol for that company.
    It does this by using the Yahoo Finance API to search for a given company, then returning
    the first result's ticker symbol.
    """
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}
    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()
    company_code = data['quotes'][0]['symbol']
    return company_code

@ray.remote
def execute_trades(trades):
    profile = r.profiles.load_account_profile()
    holdings = r.build_holdings()
    owned_stocks = list(holdings.keys())
    buying_power = float(profile['buying_power'])
    for trade in tqdm(trades):
        # update the tqdm message with trade details
        tqdm.write(f'{trade[1]} {trade[0]}')
        company, action = trade
        try:
            company_ticker = getTicker(company)
        except Exception as e:
            continue
        if company_ticker not in owned_stocks and action == 'SELL':
            continue
        position_size = min(buying_power * 0.01, buying_power)
        if action == 'BUY':
            time.sleep(4)
            trader(company_ticker, action='BUY')
            time.sleep(random.randint(1, 4))
        elif action == 'SELL':
            if company not in owned_stocks:
                continue
            time.sleep(4)
            r.orders.order_sell_stop_loss(symbol=company, quantity=0.95 * position_size, stopPrice=0.95 * position_size, timeInForce='gfd')

@ray.remote
def get_articles_from_news_source(news_source):
    # use newspaper3k to get articles
    paper = newspaper.build(news_source, memoize_articles=False)
    articles = []
    for article in paper.articles:
        try:
            tqdm.write(Fore.BLUE + str(article.title))
        except:
            pass
        try:
            article.download()
            article.parse()
            articles.append(article.text)
        except:
            continue
    return articles

@ray.remote
def check_for_stop_loss_orders():
    while True:
        try:
            cryptos = r.crypto.get_crypto_positions()
            for crypto in cryptos:
                current_price = r.crypto.get_crypto_quote(crypto['currency']['code'], info='mark_price')
                if current_price is None or not isinstance(current_price, (int, float)):
                    print(f"Error getting current price for {crypto['currency']['code']}")
                    continue
                stop_loss_price = float(current_price) * 0.99
                open_orders = r.orders.get_all_open_crypto_orders()
                for order in open_orders:
                    if order['side'] == 'sell' and order['price'] == stop_loss_price:
                        r.orders.cancel_crypto_order(order['id'])
                r.orders.order_sell_crypto_limit(crypto['currency']['code'], crypto['quantity'], stop_loss_price)
            time.sleep(60)
        except Exception as e:
            print(e)
            time.sleep(60)
            continue

@ray.remote
def remove_duplicates(articles):
    unique_articles = []
    for article in tqdm(articles):
        try:
            tqdm.write(Fore.BLUE + str(article.title))
        except:
            pass
        if not any(fuzz.ratio(article, unique_article) > 80 for unique_article in unique_articles):
            unique_articles.append(article)
    return unique_articles

import newspaper
@ray.remote
def get_articles_from_news_sources(news_sources):
    # using newspaper3k to get articles
    all_articles = []
    news_dict = {}
    # build the newspapers first
    for news_source in news_sources:
        articles = newspaper.build(news_source, memoize_articles=False).articles
        for article in articles:
            try:
                news_dict[news_source][article.title] = article
            except:
                news_dict[news_source] = {article.title: article}
    # then get the articles
    for news_source in news_dict:
        for article in news_dict[news_source]:
            try:
                tqdm.write(Fore.BLUE + str(article.title))
            except:
                pass
            try:
                article.download()
                article.parse()
                all_articles.append(article.text)
            except:
                continue
    return all_articles





# Using Ray to parallelize the execution of trades
# Main loop
news_sources = [
    'https://www.drudge.com/',
    'https://www.bloomberg.com/',
    'https://www.marketwatch.com/',
    'https://www.fool.com/',
    'https://www.businessinsider.com/',
    'https://www.axios.com/',
    'https://www.politico.com/',
    'https://www.theatlantic.com/',
    'https://www.newyorker.com/',
    'https://www.economist.com/',
    'https://www.usnews.com/',
    'https://www.nationalreview.com/',
    'https://www.motherjones.com/',
    'https://www.salon.com/',
    'https://www.slate.com/',
    'https://www.vox.com/',
    'https://www.thehill.com/',
    'https://www.vice.com/',
    'https://www.buzzfeed.com/',
    'https://www.dailykos.com/',
]

crypto_news_sources = [
    'https://cointelegraph.com/tags/robinhood',
    'https://finance.yahoo.com/news/',
    'https://finance.yahoo.com/cryptocurrencies/',
    'https://www.coindesk.com/',
    'https://cryptonews.com/',
    'https://crypto.news/',
    'https://www.cryptoglobe.com/latest/',
    'https://www.cryptocraft.com/',
    'https://www.cryptopolitan.com/',
    'https://www.reuters.com/technology/cryptocurrency',
    'https://www.coinspeaker.com/',
    'https://cryptopotato.com/crypto-news/',
    'https://www.cryptocointrade.com/',
    'https://www.bloomberg.com/crypto'
]

while True:
    # Get news sources
    # update the influence dict with the sentiment scores of the articles from the news sources
    for news_source in news_sources:
        articles = get_articles_from_news_source.remote(news_source)
        articles = ray.get(articles)
        articles = remove_duplicates.remote(articles)
        articles = ray.get(articles)
        sentiment_scores = get_sentiment_scores.remote(articles)
        sentiment_scores = ray.get(sentiment_scores)
        for company in sentiment_scores:
            if company in influence_dict:
                influence_dict[company] += sentiment_scores[company]
            else:
                influence_dict[company] = sentiment_scores[company]

    # Get crypto news sources
    # update the influence dict with the sentiment scores of the articles from the news sources
    for news_source in crypto_news_sources:
        articles = get_articles_from_news_source.remote(news_source)
        articles = ray.get(articles)
        articles = remove_duplicates.remote(articles)
        articles = ray.get(articles)
        sentiment_scores = get_sentiment_scores.remote(articles)
        sentiment_scores = ray.get(sentiment_scores)
        for company in sentiment_scores:
            if company in influence_dict:
                influence_dict[company] += sentiment_scores[company]
            else:
                influence_dict[company] = sentiment_scores[company]

    # Get the top 5 companies with the highest sentiment scores
    top_5_companies = sorted(influence_dict, key=influence_dict.get, reverse=True)[:5]
    # Get the top 5 companies with the lowest sentiment scores
    bottom_5_companies = sorted(influence_dict, key=influence_dict.get, reverse=False)[:5]
    # Get the top 5 companies with the highest sentiment scores
    top_5_crypto_companies = sorted(influence_dict, key=influence_dict.get, reverse=True)[:5]
    # Get the top 5 companies with the lowest sentiment scores
    bottom_5_crypto_companies = sorted(influence_dict, key=influence_dict.get, reverse=False)[:5]
