import requests
import time
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from robin_stocks import robinhood as r
from tqdm import tqdm
# import beautiful soup
from bs4 import BeautifulSoup
from ratelimit import limits, sleep_and_retry
# Set up Robinhood credentials
# get username and password from the secrets.json file
import json
with open('secrets.json') as f:
    secrets = json.load(f)

username = secrets['username']
password = secrets['password']
login = r.login(username, password)

# Initialize sentiment analyzer and Spacy NER
sia = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

def get_company_names(text):
    # Use Spacy's NER to identify company names
    doc = nlp(text)
    company_names = [ent.text for ent in doc.ents if ent.label_ == 'ORG']

    return company_names

def calculate_sentiment(text):
    # Calculate sentiment score using NLTK's SentimentIntensityAnalyzer
    sentiment = sia.polarity_scores(text)
    sentiment_score = sentiment['compound']

    return sentiment_score

def get_news_sentiment():
    # Fetch news articles from Newspaper4k using the company names
    # and calculate sentiment for each article
    sentiment_scores = {}
    url = 'https://www.drudge.com/'
    response = requests.get(url)
    html = response.text
    company_names = get_company_names(html)

    for company in company_names:
        # Fetch news articles using requests
        url = f'https://news.google.com/rss/search?q={company}'
        response = requests.get(url)
        html = response.text
        # get text from html using beautifulsoup
        text = BeautifulSoup(html, 'html.parser').get_text()
        # Calculate sentiment score for each article using calculate_sentiment() function
        sentiment_score = calculate_sentiment(text)

        # Store sentiment score for the company
        sentiment_scores[company] = sentiment_score

    return sentiment_scores

@sleep_and_retry()
def generate_trades():
    # Get sentiment scores for news articles
    sentiment_scores = get_news_sentiment()

    # Determine trading decisions based on sentiment scores
    trades = []
    for company, sentiment_score in sentiment_scores.items():
        if sentiment_score > 0.2:
            # Generate buy signal for companies with positive sentiment
            trades.append((company, 'BUY'))
        elif sentiment_score < -0.2:
            # Generate sell signal for companies with negative sentiment
            trades.append((company, 'SELL'))

    return trades

@sleep_and_retry()
def execute_trades():
    # Generate trading signals based on sentiment scores
    trades = generate_trades()

    # Get buying power
    profile = r.profiles.load_account_profile()
    buying_power = float(profile['buying_power'])

    # Execute trades using Robinhood
    for trade in tqdm(trades):
        company, action = trade
        # Calculate position size (not exceeding 1% of buying power)
        position_size = min(buying_power * 0.01, buying_power)
        # position size is in dollars, convert to number of shares
        position_size = position_size / float(r.stocks.get_latest_price(company)[0])
        if action == 'BUY':
            # Place buy order
            print(f"Buying {position_size} shares of {company}")
            r.orders.order_buy_market(company, position_size)
            # Set trailing stop loss (1%)
            print(f'--> Setting trailing stop loss for {company}, at 1%')
            r.orders.order_sell_trailing_stop(symbol=company,
                                              quantity=position_size,
                                              trail_percent=1)
        elif action == 'SELL':
            # Place sell order with stop loss (5%)
            print(f"Selling {position_size} shares of {company}")
            r.orders.order_sell_stop_loss(symbol=company,
                                          quantity=position_size,
                                          stop_price=position_size*0.95)

    return trades

# Execute trading algorithm at high frequency but low volume
# ratelimit to avoid getting banned by Robinhood
import time
from ratelimit import RateLimitException, sleep_and_retry, limits

while True:
    trades = execute_trades()
    # Sleep for a short period before scanning for new signals
    time.sleep(60)
