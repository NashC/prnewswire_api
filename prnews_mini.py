#prnews_mini.py

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from pymongo import MongoClient
from time import time

'''
- topic clustering
- NMF
- how topics vary or group by industry (get dummies style)
-sentiment analysis: spacy.io (https://spacy.io/), nltk.sentiment (http://www.nltk.org/) (http://www.nltk.org/api/nltk.sentiment.html#module-nltk.sentiment.sentiment_analyzer)
'''

def mongo_to_df(db, collection):
	connection = MongoClient()
	db = connection[db]
	input_data = db[collection]
	df = pd.DataFrame(list(input_data.find()))
	return df

def prep_df(df):
	df.drop_duplicates(subset=['article_id'], inplace=True)
	df['release_text'] = df['release_text'].apply(lambda x: x.lstrip('\n'))
	df['release_text'] = df['release_text'].apply(lambda x: x.rstrip('\n'))
	df['release_text'] = df['release_text'].apply(lambda x: x.replace('\n', ' '))
	df['release_text'] = df['release_text'].apply(lambda x: x.replace(u'\xa0', u' '))

	return df

df_orig = mongo_to_df('press', 'big_2')
df = prep_df(df_orig)
release_texts = df['release_text']

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(release_texts)

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
tf = tf_vectorizer.fit_transform(release_texts)

# nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
nmf = NMF(n_components=n_topics, random_state=1)
nmf.fit(tfidf)

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

