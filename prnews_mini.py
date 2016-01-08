#prnews_mini.py

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.grid_search import GridSearchCV
from nltk import tokenize
from nltk.corpus import stopwords
from pymongo import MongoClient
from time import time
import re
from textblob import TextBlob
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score


def mongo_to_df(db, collection):
	connection = MongoClient('127.0.0.1', 27017)
	db = connection[db]
	input_data = db[collection]
	df = pd.DataFrame(list(input_data.find()))
	return df

def make_dummies(df, col, prefix):
	s = df[col]
	dummies = pd.get_dummies(s.apply(pd.Series), prefix=prefix, prefix_sep='_').sum(level=0, axis=1)
	result = pd.concat([df, dummies], axis=1)
	return result

def get_cities(text):
	city_date = text.split('PRNewswire')[0].split()
	temp = []
	for i, word in enumerate(city_date[:-1]):
		if word.isupper():
			if city_date[i+1].isupper():
				temp.append(' '.join((city_date[i], city_date[i+1])))
			elif city_date[i-1].isupper():
				continue
			else:
				temp.append(word)
	return temp

def lemmatize(doc_text):
	# doc_text = doc_text.encode('utf-8')
	blob = TextBlob(doc_text)
	temp = []
	for word in blob.words:
		temp.append(word.lemmatize())
	result = ' '.join(temp)
	return result

def prep_text(df):
	df.drop_duplicates(subset=['article_id'], inplace=True)
	df['city'] = df['release_text'].apply(lambda x: get_cities(x))
	df['release_text'] = df['release_text'].apply(lambda x: re.sub(r'\(?(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)\)?', '', x, flags=re.M))
	df['release_text'] = df['release_text'].apply(lambda x: lemmatize(x))
	return df

def print_top_words(model, feature_names, n_top_words):
	for topic_idx, topic in enumerate(model.components_):
		print("Topic #%d:" % topic_idx)
		print(" ".join([feature_names[i]
						for i in topic.argsort()[:-n_top_words - 1:-1]]))
	pass

new_stop_words = ['ha', "\'s", 'tt', 'ireach', "n\'t", 'wo']
def make_stop_words(new_words_list):
	tfidf_temp = TfidfVectorizer(stop_words='english')
	stop_words = tfidf_temp.get_stop_words()
	result = list(stop_words) + new_words_list
	return result

def row_normalize_tfidf(sparse_matrix):
	return normalize(sparse_matrix, axis=1, norm='l1')

def get_topics(n_components=10, n_top_words=15, print_output=True):
	custom_stop_words = make_stop_words(new_stop_words)
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=custom_stop_words)
	tfidf = tfidf_vectorizer.fit_transform(release_texts)
	tfidf = row_normalize_tfidf(tfidf)

	nmf = NMF(n_components=n_components, random_state=1)
	# nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
	nmf.fit(tfidf)
	W = nmf.transform(tfidf)
	

	if print_output:
		print("\nTopics in NMF model:")
		tfidf_feature_names = tfidf_vectorizer.get_feature_names()
		print_top_words(nmf, tfidf_feature_names, n_top_words)
	
	return tfidf, nmf, W


y_mse = [4.1586519190079731e-07,
 4.0806056611869579e-07,
 4.0635976762741562e-07,
 4.0469315509174745e-07,
 4.031626753591851e-07,
 4.0198547133833227e-07,
 4.0069371691965978e-07,
 3.9989096055799519e-07,
 3.9873993330429839e-07,
 3.9755992608438446e-07,
 3.9668909965925691e-07,
 3.9565356967902645e-07,
 3.9478109096903595e-07,
 3.9395900472461631e-07,
 3.9306251341616297e-07,
 3.922508964184518e-07,
 3.9139907529199026e-07,
 3.9058720813092185e-07,
 3.8985442673013608e-07,
 3.8906441983276564e-07,
 3.8829184088224553e-07,
 3.8795241871128741e-07,
 3.8670916159757251e-07,
 3.8630316709902775e-07,
 3.8551159536125117e-07,
 3.8529296023215857e-07,
 3.8405309135809741e-07,
 3.8349524545279573e-07,
 3.8312979033889216e-07,
 3.8214679236464401e-07,
 3.8163433071433644e-07,
 3.812476213782855e-07,
 3.8037440645891213e-07,
 3.8003752994030476e-07,
 3.7912925338958192e-07,
 3.7869750967188198e-07,
 3.7801548354496874e-07,
 3.7721730759096696e-07,
 3.7667911984978913e-07,
 3.76411057692006e-07,
 3.7596695262405878e-07,
 3.7503280027874852e-07,
 3.74409880730487e-07,
 3.7385335317775083e-07,
 3.7370292291722861e-07,
 3.7324109586296963e-07,
 3.7239348121954796e-07,
 3.7220875451942766e-07,
 3.7157594400522471e-07,
 3.7092184827551988e-07]
x_range = np.arange(1,51)

def grid_search_nmf_ncomponents(tfidf, low, high):
	tfidf_dense = tfidf.toarray()
	mse_min = 99
	mse_min_ncomponents = -1
	for i in xrange(low, high + 1):
		print 'Fitting n_components = %d ...' %i
		nmf_temp = NMF(n_components=i, random_state=1)
		# cv = cross_val_score(nmf_temp, tfidf, scoring='mean_squared_error', cv=5)
		nmf_temp.fit(tfidf)
		W = nmf_temp.transform(tfidf)
		H = nmf_temp.components_
		tfidf_pred = np.dot(W, H)
		mse_temp = mean_squared_error(tfidf_dense, tfidf_pred)
		y_mse.append(mse_temp)
		x_range.append(i)
		print 'MSE of n_components = %d: %.10f' %(i, mse_temp)
		print '-------------------------------'
		if mse_temp < mse_min:
			mse_min = mse_temp
			mse_min_ncomponents = i
	return mse_min_ncomponents

def sum_dummie_counts(df):
	for col in df.columns:
		try:
			print col, sum(df[col])
		except:
			pass

def textblob_sentiment(text):
	blob = TextBlob(text)
	polarity = blob.sentiment.polarity
	subjectivity = blob.sentiment.subjectivity
	return (polarity, subjectivity)

def add_sentiment_to_df(df):
	df['polarity'] = df['release_text'].apply(lambda x: textblob_sentiment(x)[0])
	df['subjectivity'] = df['release_text'].apply(lambda x: textblob_sentiment(x)[1])
	return df

nmf_grid = {'n_components': np.arange(3,25)}
def grid_search(est, grid, train_data):
    grid_cv = GridSearchCV(est, grid, n_jobs=-1, verbose=True,
                           scoring='mean_squared_error').fit(train_data)
    return grid_cv


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

df_orig = mongo_to_df('press', 'test_master_1')
subject_dict = make_subject_dict()
industry_dict = make_industry_dict()

df = prep_text(df_orig)
df = make_dummies(df, 'industry', 'ind')
df = make_dummies(df, 'subject', 'subj')
df = add_sentiment_to_df(df)
release_texts = df['release_text']

tfidf, nmf, W = get_topics(print_output=True)

# nmf_grid_search = grid_search(NMF(), nmf_grid, tfidf)




