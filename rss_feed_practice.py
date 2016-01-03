#rss_feed_practice.py

import pandas as pd
import numpy as np
import feedparser as fp
import requests
import bs4
import json
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import datetime as dt

d = fp.parse('http://www.prnewswire.com/rss/financial-services/venture-capital-news.rss')

db_client = MongoClient()
db = db_client['rss']
table = db['vc_news']

def single_query(link):
    response = requests.get(link)
    if response.status_code != 200:
        print 'WARNING', response.status_code
    else:
        return response

link = str(d.entries[0].link)
# print link

r = requests.get(link)

html_txt = r.text

soup = bs4.BeautifulSoup(html_txt, 'html.parser')

print soup.find_all('p', itemprop='articleBody')