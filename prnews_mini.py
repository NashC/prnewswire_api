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
import re

'''
- topic clustering
- NMF
- how topics vary or group by industry/subject/geography (get dummies style)
-sentiment analysis: spacy.io (https://spacy.io/), nltk.sentiment (http://www.nltk.org/) (http://www.nltk.org/api/nltk.sentiment.html#module-nltk.sentiment.sentiment_analyzer)
'''

def mongo_to_df(db, collection):
	connection = MongoClient()
	db = connection[db]
	input_data = db[collection]
	df = pd.DataFrame(list(input_data.find()))
	return df

def make_dummies(df, col, prefix):
	s = df[col]
	dummies = pd.get_dummies(s.apply(pd.Series), prefix=prefix, prefix_sep='_').sum(level=0, axis=1)
	result = pd.concat([df, dummies], axis=1)
	return result

def prep_df(df):
	df.drop_duplicates(subset=['article_id'], inplace=True)

	df['release_text'] = df['release_text'].apply(lambda x: x.lstrip('\n'))
	df['release_text'] = df['release_text'].apply(lambda x: x.rstrip('\n'))
	df['release_text'] = df['release_text'].apply(lambda x: x.replace('\n', ' '))
	df['release_text'] = df['release_text'].apply(lambda x: x.replace(u'\xa0', u' '))
	df['release_text'] = df['release_text'].apply(lambda x: re.sub('^\(?https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE))
	df['release_text'] = df['release_text'].apply(lambda x: re.sub('^\(?www\..*[\r\n]*', '', x, flags=re.MULTILINE))
	return df

# df_orig = mongo_to_df('press', 'big_2')
# df_orig = mongo_to_df('press', 'big_2_42600')
# df_orig = mongo_to_df('press', 'big_2_98600')
df_orig = mongo_to_df('press', 'test_master_1')
df = prep_df(df_orig)
df = make_dummies(df, 'industry', 'ind')
df = make_dummies(df, 'subject', 'subj')

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
    pass

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

def sum_dummie_counts(df):
	for col in df.columns:
		try:
			print col, sum(df[col])
		except:
			pass

def make_subject_dict():
	subject_list = [
	[u'ACC', u'Accounting News, Issues'],
	[u'TNM', u'Acquisitions, Mergers, Takeovers'],
	[u'BCY', u'Bankruptcy'],
	[u'RTG', u'Bond/Stock Ratings'],
	[u'CON', u'Contracts'],
	[u'DIV', u'Dividends'],
	[u'ERN', u'Earnings'],
	[u'ERP', u'Earnings Projects or Forecasts'],
	[u'ECO', u'Economic News, Trends and Analysis'],
	[u'FNC', u'Financing Agreements'],
	[u'JVN', u'Joint Ventures'],
	[u'LIC', u'Licensing/Marketing Agreements'],
	[u'PDT', u'New Products/Services'],
	[u'OFR', u'Offerings'],
	[u'DSC', u'Oil/Gas Discoveries'],
	[u'OTC', u'OTC/SmallCap IRW'],
	[u'PER', u'Personnel Announcements'],
	[u'RLE', u'Real Estate Transactions'],
	[u'RCN', u'Restructuring/Recapitalizations'],
	[u'SLS', u'Sales Reports'],
	[u'SRP', u"Shareholders' Rights Plans"],
	[u'LEG', u'Federal and State Legislation'],
	[u'EXE', u'Federal Executive Branch, Agency News'],
	[u'CPN', u'Political Campaigns'],
	[u'LBR', u'Labor/Union news'],
	[u'BLK', u'African-American News'],
	[u'HSP', u'Hispanic-Oriented News'],
	[u'LAW', u'Legal Issues'],
	[u'AWD', u'Awards'],
	[u'NPT', u'Not for Profit'],
	[u'TDS', u'Tradeshow News'],
	[u'CCA', u'Conference Call Announcements'],
	[u'CHI', u'Children-Related News'],
	[u'WOM', u'Women-related News'],
	[u'VEN', u'Venture Capital'],
	[u'BFA', u'Broadcast Feed Announcement'],
	[u'ASI', u'Asian-Related News'],
	[u'EGV', u'European Government'],
	[u'MAV', u'Media Advisory/Invitation'],
	[u'SVY', u'Surveys, Polls & Research'],
	[u'INO', u'Investments Opinions'],
	[u'ZHA', u'Xinhua'],
	[u'FOR', u'Foreign policy/International affairs'],
	[u'POL', u'Domestic Policy'],
	[u'TRD', u'Trade Policy'],
	[u'REL', u'Religion'],
	[u'STS', u'Stock Split'],
	[u'PET', u'Animals/Pets'],
	[u'TRI', u'Clinical Trials/Medical Discoveries'],
	[u'RCY', u'Conservation/Recycling'],
	[u'CSR', u'Corporate Social Responsibility'],
	[u'FDA', u'FDA Approval'],
	[u'DIS', u'Handicapped/Disabled'],
	[u'LGB', u'Lesbian/Gay/Bisexual'],
	[u'NTA', u'Native American'],
	[u'PLW', u'Patent Law'],
	[u'RCL', u'Product Recalls'],
	[u'PSF', u'Public Safety'],
	[u'SCZ', u'Senior Citizens'],
	[u'SBS', u'Small Business Services'],
	[u'STP', u'U.S. State Policy News'],
	[u'VET', u'Veterans'],
	[u'VDM', u'MultiVu Video'],
	[u'ADM', u'MultiVu Audio'],
	[u'PHM', u'MultiVu Photo'],
	[u'BCM', u'Broadcast Minute'],
	[u'CXP', u'Corporate Expansion'],
	[u'ENI', u'Environmental Issues'],
	[u'ENP', u'Environmental Policy'],
	[u'SRI', u'Socially Responsible Investing'],
	[u'VNR', u'Video News Releases'],
	[u'ANW', u'Animal Welfare'],
	[u'AVO', u'Advocacy Group Opinion'],
	[u'OBI', u'Obituaries'],
	[u'FEA', u'Features']]

	subject_dict = {x[0]:x[1] for x in subject_list}
	return subject_dict
subject_dict = make_subject_dict()

def make_industry_dict():
	ind_list = [
	[u'ADV', u'Advertising '],
	[u'ARO', u'Aerospace/Defense'],
	[u'AGR', u'Agriculture'],
	[u'AIR', u'Airlines/Aviation'],
	[u'ART', u'Art'],
	[u'AUT', u'Automotive'],
	[u'FIN', u'Banking/Financial Services'],
	[u'BIO', u'Biotechnology'],
	[u'BKS', u'Books'],
	[u'CHM', u'Chemical'],
	[u'CPR', u'Computer/ Electronics'],
	[u'NET', u'Networks'],
	[u'HRD', u'Computer Hardware'],
	[u'STW', u'Computer Software'],
	[u'CST', u'Construction/Building'],
	[u'CSE', u'Consumer Electronics'],
	[u'EDU', u'Education'],
	[u'EPM', u'Electronics Performance Measurement'],
	[u'ECM', u'Electronic Commerce'],
	[u'ENT', u'Entertainment'],
	[u'ENV', u'Environmental Products & Services'],
	[u'FAS', u'Fashion'],
	[u'FLM', u'Film and Motion Picture'],
	[u'FOD', u'Food & Beverages'],
	[u'CNO', u'Gambling/Casinos'],
	[u'HEA', u'Health Care/Hospitals'],
	[u'HOU', u'Household/Consumer/Cosmetics'],
	[u'INS', u'Insurance'],
	[u'ITE', u'Internet Technology'],
	[u'LEI', u'Leisure & Tourism'],
	[u'MAC', u'Machinery'],
	[u'MAG', u'Magazines'],
	[u'MAR', u'Maritime/Shipbuilding'],
	[u'MTC', u'Medical/Pharmaceuticals'],
	[u'MNG', u'Mining/Metals'],
	[u'MLM', u'Multimedia/Internet'],
	[u'MUS', u'Music'],
	[u'MFD', u'Mutual Funds'],
	[u'OFP', u'Office Products'],
	[u'OIL', u'Oil/Energy'],
	[u'PAP', u'Paper/Forest Products/Containers'],
	[u'PEL', u'Peripherals'],
	[u'PUB', u'Publishing/Information Services'],
	[u'RAD', u'Radio'],
	[u'RLT', u'Real Estate'],
	[u'REA', u'Retail'],
	[u'RST', u'Restaurants'],
	[u'SPT', u'Sports'],
	[u'SUP', u'Supermarkets'],
	[u'SPM', u'Supplementary Medicine'],
	[u'TLS', u'Telecommunications Industry'],
	[u'TVN', u'Television'],
	[u'TEX', u'Textiles'],
	[u'TOB', u'Tobacco'],
	[u'TRN', u'Transportation/Trucking/Railroad'],
	[u'TRA', u'Travel'],
	[u'UTI', u'Utilities'],
	[u'Feature', u'Features'],
	[u'HTS', u'High Tech Security'],
	[u'ECP', u'Electronic Components'],
	[u'EDA', u'Electronic Design Automation'],
	[u'SEM', u'Semiconductors'],
	[u'HED', u'Higher Education'],
	[u'ALC', u'Beers, Wines and Spirits'],
	[u'BIM', u'Biometrics'],
	[u'GAM', u'Electronic Gaming'],
	[u'HMS', u'Homeland Security'],
	[u'IDC', u'Infectious Disease Control'],
	[u'MEN', u'Mobile Entertainment'],
	[u'NAN', u'Nanotechnology'],
	[u'WRK', u'Workforce Management/Human Resources'],
	[u'AIF', u'Air Freight'],
	[u'ALT', u'Alternative Energies'],
	[u'ANW', u'Animal Welfare'],
	[u'ATL', u'Amusement Parks and Tourist Attractions'],
	[u'BEV', u'Beverages'],
	[u'BRI', u'Bridal Services'],
	[u'CPC', u'Cosmetics and Personal Care'],
	[u'CRL', u'Commercial Real Estate'],
	[u'DEN', u'Dentistry'],
	[u'ENS', u'Environmental Products & Services'],
	[u'EUT', u'Electrical Utilities'],
	[u'FRN', u'Furniture and Furnishings'],
	[u'GAS', u'Gas'],
	[u'HHP', u'Household Products'],
	[u'HIN', u'Health Insurance '],
	[u'HMI', u'Home Improvements'],
	[u'HRT', u'Hotels and Resorts'],
	[u'HSC', u'Home Schooling'],
	[u'HVA', u'HVAC'],
	[u'JWL', u'Jewelry'],
	[u'MCT', u'Machine Tools, Metalworking and Metallurgy'],
	[u'MEQ', u'Medical Equipment'],
	[u'MIN', u'Mining'],
	[u'MNH', u'Mental Health'],
	[u'NAB', u'Non-Alcoholic Beverages'],
	[u'ORF', u'Organic Food'],
	[u'ORL', u'Overseas Real Estate (non-US) '],
	[u'OUT', u'Outsourcing Businesses'],
	[u'PAV', u'Passenger Aviation'],
	[u'PHA', u'Pharmaceuticals'],
	[u'PRM', u'Precious Metals'],
	[u'RFI', u'RFID (Radio Frequency ID) Applications & Tech'],
	[u'RIT', u'Railroads & Intermodal Transporation'],
	[u'RRL', u'Residential Real Estate'],
	[u'SMD', u'Social Media'],
	[u'SPE', u'Sports Equipment & Accessories'],
	[u'SSE', u'Sporting Events'],
	[u'SWB', u'Semantic Web'],
	[u'TCS', u'Telecommunications Carriers and Services'],
	[u'TEQ', u'Telecommunications Equipment'],
	[u'TRT', u'Trucking and Road Transportation'],
	[u'VIP', u'VoIP (Voice over Internet Protocol)'],
	[u'WEB', u'Web site'],
	[u'WIC', u'Wireless Communications'],
	[u'WUT', u'Water Utilities'],
	[u'GRE', u'Green Technology'],
	[u'OTC', u'OTC/SmallCap'],
	[u'SRI', u'Socially Responsible Investing'],
	[u'TOY', u'Toys'],
	[u'BRD', u'Broadcast Technology']
	]
	ind_dict = {x[0]:x[1] for x in ind_list}
	return ind_dict