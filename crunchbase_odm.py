#crunchbase_odm

import json
import pandas as pd
from pprint import pprint



def make_orgs_dict():
	with open('cb_odm/organizations.json') as data_file:
		data = json.load(data_file)
	org_dicts = data['root']
	result = {}
	for org in org_dicts:
		if 'name' in org.keys() and org['primary_role'] == 'company':
			if type(org['name']) == int:
				name = str(org['name'])
			else:
				name = org['name'].encode('utf-8', 'ignore').lower()
			result[name] = {}
			for k in org.iterkeys():
				if type(org[k]) == int:
					result[name][str(k).lower()] = str(org[k])
				else:
					result[name][str(k).lower()] = org[k].encode('utf-8', 'ignore').lower()
	return result

csv_file = 'cb_odm_csv/organizations.csv'

def make_orgs_df():
	df = pd.read_csv('cb_odm_csv/organizations.csv')
	df_filt = df[(df['primary_role'] == 'company') & (df['location_country_code'] == 'USA') & (df['location_region'].isin(['California', 'New York', 'Washington', 'Colorado', 'Texas', 'Connecticut', 'Massachusetts', 'New Jersey']))]
	return df_filt


