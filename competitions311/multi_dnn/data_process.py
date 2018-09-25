#encoding:utf-8
import sys, pickle, os, random
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
tag2label = {"90063345": 0,  
             "89950166": 1, "89950167": 2,
             "99999828":3, "89016252":4,
             "99104722":5, "90109916":6,
             "89950168":7, "99999827":8,
             "99999826":9, "90155946":10,
             "99999830":11, "99999825":12,
             "89016253":13, "89016259":14
             }
ID_COLUMN_NAME = 'user_id'
LABEL_COLUMN_NAME = 'predict'

def read_corpus(corpus_path):
	data = []
	with open(corpus_path) as fr:
		lines = fr.readlines()
	feat_ = []
	tag_ = []
	for line in lines:
		terms = line.strip().split('\t')
		for i in range(len(terms)-1):
			feat_.append(terms[i])
		tag_.append(terms[-1])

		data.append((feat_,tag_))
		feat_ = []
		tag_ = []
	return data
def read_corpus_test(corpus_path):
	data = []
	with open(corpus_path,'r') as fr:
		lines = fr.readlines()
	feat_ = []
	for line in lines:
		terms = line.strip().split('\t')
		for term in terms:
			feat_.append(term)

		data.append(feat_)
		feat_ = []
	return data

def batch_yield(data, batch_size, tag2label):
	random.shuffle(data)

	feats, labels = [], []
	for (feat_, tag_) in data:
		label_ = [tag2label[tag] for tag in tag_]

		if len(feats) == batch_size:
			yield feats, labels
			feats, labels = [], []
		feats.append(feat_)
		labels.append(label_[0])
	if len(feats) != 0:
		yield feats, labels

def batch_yield_test(data, batch_size, tag2label):
	feats = []
	for feat_ in data:
		if len(feats) == batch_size:
			yield feats
			feats = []
		feats.append(feat_)
	if len(feats) != 0:
		yield feats

def save_result(data, path):
	csv_input = os.path.join('.',path,'test.csv')
	csv_output = os.path.join('.',path,'submit_result.csv')
	df_test = pd.read_csv(csv_input)
	print('====shape df_test====',df_test.shape)
	user_id_list = np.array(df_test[ID_COLUMN_NAME]).tolist()
	if len(user_id_list) != len(data):
		print('test_data len error')
		print('====len user_id_list===',len(user_id_list))
		print('====data===',len(data))
	result = {ID_COLUMN_NAME:user_id_list,
				LABEL_COLUMN_NAME:data}
	submit_csv = pd.DataFrame(result)
	submit_csv.to_csv(csv_output, index = False)





