import configparser
import feedparser
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
import logging
import requests
import seaborn as sns
import warnings
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from newspaper import Article
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dateutil import parser
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
#import EC and By
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# set up a logging file for the text analysis in logs/ named text_analysis.log and set the logging level to INFO
logging.basicConfig(filename='logs/text_analysis.log', level=logging.INFO)


warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

config = configparser.ConfigParser()
config.read('config/credentials.ini')
coins = config['trading']['coins'].split(',')
robinhood_news_urls = {}
reddit_rss_feeds = {}
coinbase_rss_feeds = {}
coin_telegraph_rss_feeds = {}
# login elements for Robinhood
password_field_css_selector = config['robinhood']['password_selector_main_page']
password_field_id = config['robinhood']['password_field_id']
username_field_css_selector = config['robinhood']['username_selector_main_page']
login_button_css_selector = config['robinhood']['login_button_css_selector']

for coin in coins:
    coin = coin.strip().lower()
    robinhood_news_urls[coin] = config['robinhood_crypto_news_urls'][str(coin)]
    coinbase_rss_feeds[coin] = config['coinbase_rss_feeds'][str(coin)]
    coin_telegraph_rss_feeds[coin] = config['coin_telegraph_rss_feeds'][str(coin)]
    reddit_rss_feeds[coin] = config['reddit_rss_feeds'][str(coin)]

sentiment_analyzer = SentimentIntensityAnalyzer()
sentiment = []
bullish_bearish = []
topic = []

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# start a selenium driver to scrape Robinhood Crypto News
options = Options()
chrome_driver_path = 'config/chromedriver'
driver = webdriver.Chrome(chrome_driver_path, options=options)
# open RobinHoods login page
driver.get('https://robinhood.com/login')


def standardize_date(unstructured_date):
    dt = parser.parse(unstructured_date)
    return dt.strftime('%Y-%m-%dT%H:%M:%S%z')

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    logging.info(f'Preprocessed text: {len(words)} words; most common: {pd.Series(words).value_counts().head(5).index.tolist()}')
    return ' '.join(words)

def scrape_robinhood_article(url):
    try:
        logging.info(f"Scraping the article at {url}")
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            article_text = soup.find("div", {"class": "article__content"}).get_text()
            return article_text
        else:
            return None
    except Exception as e:
        logging.error(f"Error scraping the article at {url}: {e}")
        return None


def scrape_robinhood_news_main_feed(url):
    # don't quit the driver after scraping, so we can reuse it. This is useful for scraping multiple pages. Also, user will need to log in to Robinhood once.
    # options = Options()
    # chrome_driver_path = 'config/chromedriver'
    # options.add_argument('--headless')
    # driver = webdriver.Chrome(chrome_driver_path, options=options)


    links = []

    try:
        driver.get(url)
        for i in range(10):
            try:
                # div = driver.find_element_by_id(f"sdp-news-row-{i}")
                # use WebDriverWait to wait for the element to load and By.class to find the element
                div = WebDriverWait(driver, 120).until(EC.presence_of_element_located((By.CLASS_NAME, f"sdp-news-row-{i}")))
                links.append(div.find_element_by_tag_name('a').get_attribute('href'))
            except Exception as e:
                logging.error(f"Error scraping the article at {url}: {e}")
                return []
        logging.info(f"Scraped {len(links)} links from Robinhood Crypto News")

    except Exception as e:
        logging.error(f"Error accessing {url} with Selenium: {e}")
        return []


    return links


def sentiment_and_topic_analysis(processed_text):
    sentiment_score = sentiment_analyzer.polarity_scores(processed_text)['compound']
    sentiment.append(sentiment_score)

    bullish_words = ['buy', 'bullish', 'long', 'up']
    bearish_words = ['sell', 'bearish', 'short', 'down']
    num_bullish = len(re.findall(r'\b(?:' + '|'.join(bullish_words) + r')\b', processed_text))
    num_bearish = len(re.findall(r'\b(?:' + '|'.join(bearish_words) + r')\b', processed_text))

    if num_bullish > num_bearish:
        bullish_bearish.append('Bullish')
    elif num_bearish > num_bullish:
        bullish_bearish.append('Bearish')
    else:
        bullish_bearish.append('Neutral')

    vectorizer = CountVectorizer(max_df=0.50, min_df=0.02, max_features=1000, stop_words='english')
    tf = vectorizer.fit_transform([processed_text])
    tf_feature_names = vectorizer.get_feature_names_out()
    num_topics = 1

    lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online',
                                            random_state=42, n_jobs=-1)
    lda_model.fit(tf)

    topic_words = [tf_feature_names[i] for i in lda_model.components_[0].argsort()[:-11:-1]]
    topic.append(', '.join(topic_words))

    return sentiment_score, num_bullish, num_bearish, topic_words

def process_feeds(rss_feeds):
    with alive_bar(len(rss_feeds.values())) as bar:
        for rss_url in rss_feeds.values():
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                bar.text(f"Processing {entry.title}")
                try:
                    if (datetime.now() - datetime.strptime(standardize_date(entry.published), '%Y-%m-%dT%H:%M:%S+00:00')).total_seconds() < 14400:
                        processed_text = preprocess_text(entry.summary)
                        sentiment_and_topic_analysis(processed_text)
                except Exception as e:
                    logging.error(f"Error processing the article: {e}")
            bar()

def process_robinhood_articles(coin):
    # process robinhoods news articles on that coin
    main_page_coin = f'https://robinhood.com/crypto/{str(coin).upper()}/news'
    # find the robinhood_news_urls on that page using scrape_robinhood_news_main_feed
    robinhood_news_urls = scrape_robinhood_news_main_feed(main_page_coin)

    # set up the lists to store the sentiment, bullish_bearish, and topic
    sentiment = []
    bullish_bearish = []
    topic = []
    coin = []

    if robinhood_news_urls is not None:  # Check if the return value is not None
        print(f'Processing {str(coin).upper()} articles')
        try:
            # scrape the article at each robinhood_news_url using scrape_robinhood_article
            print(f'Scraping {len(robinhood_news_urls)} articles from Robinhood Crypto News')
            for url in robinhood_news_urls:
                print(f'>> Scraping {url}')
                article_text = scrape_robinhood_article(url)
                if article_text:
                    processed_text = preprocess_text(article_text)
                    sentiment_score, num_bullish, num_bearish, topic_words = sentiment_and_topic_analysis(processed_text)
                    logging.info(f"Sentiment score: {sentiment_score}; Bullish words: {num_bullish}; Bearish words: {num_bearish}; Topic words: {topic_words}")
                    # fill in the sentiment, bullish_bearish, and topic lists with the values returned by sentiment_and_topic_analysis
                    sentiment.append(sentiment_score)
                    if num_bullish > num_bearish:
                        bullish_bearish.append('Bullish')
                    elif num_bearish > num_bullish:
                        bullish_bearish.append('Bearish')
                    else:
                        bullish_bearish.append('Neutral')
                    topic.append(', '.join(topic_words))
                    coin.append(coin)
                else:
                    # fill in the sentiment, bullish_bearish, and topic lists with None values
                    sentiment.append(None)
                    bullish_bearish.append(None)
                    topic.append(None)
                    coin.append(None)
        except Exception as e:
            logging.error(f"Error processing the article: {e}")
            # fill in the sentiment, bullish_bearish, and topic lists with None values
            sentiment.append(None)
            bullish_bearish.append(None)
            topic.append(None)
            coin.append(None)
    else:
        logging.error(f"Error processing the article at {main_page_coin}")
        # fill in the sentiment, bullish_bearish, and topic lists with None values
        sentiment.append(None)
        bullish_bearish.append(None)
        topic.append(None)
        coin.append(None)

for coin in coins:
    coin = coin.upper().strip()
    print(f'Processing {str(coin).upper()} articles')
    process_robinhood_articles(coin)


df = pd.DataFrame({'Sentiment': sentiment, 'Bullish/Bearish': bullish_bearish,
                   'Suggestion': None, 'Topic': topic}, index=coins)
def generate_suggestion(row):
    if row['Bullish/Bearish'] == 'Bullish' and row['Sentiment'] > 0:
        return 'Buy'
    elif row['Bullish/Bearish'] == 'Bearish' and row['Sentiment'] < 0:
        return 'Sell'
    else:
        return 'Hold'

df['Suggestion'] = df.apply(generate_suggestion, axis=1)

print(df)
