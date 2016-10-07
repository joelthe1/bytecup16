import numpy as np
import pandas as pd
import sys
import copy
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.naive_bayes import GaussianNB

###
#prepare data
###

###
# 1. Load Data
###
print "\nLoading Data..."
question_path = "bytecup2016data/question_info.txt"
user_path = "bytecup2016data/user_info.txt"
invited_info_path = "bytecup2016data/invited_info_train.txt"

q_column_names = ['q_id', 'q_tag', 'q_word_seq', 'q_char_seq', 'q_no_upvotes', 'q_no_answers', 'q_no_quality_answers']
u_column_names = ['u_id','e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']
train_info_column_names = ['q_id','u_id','answered']

question_dataframe = pd.read_csv(question_path, names=q_column_names, sep = '\t')
user_dataframe = pd.read_csv(user_path, names = u_column_names, sep = '\t')
train_info_dataframe = pd.read_csv(invited_info_path, names = train_info_column_names, sep = '\t')
print "Loading Data (complete)..."

###
# 2. Prepare Vocabulary
###

print "\nPreparing Vocabulary..."
all_word_desc_list = question_dataframe['q_word_seq'].tolist() + user_dataframe['e_desc_word_seq'].tolist()
all_char_desc_list = question_dataframe['q_char_seq'].tolist() + user_dataframe['e_desc_char_seq'].tolist()
all_topics_list = question_dataframe['q_tag'].tolist() + user_dataframe['e_expert_tags'].tolist()

word_vocabulary = set([word for sent in all_word_desc_list for word in str(sent).split('/')])
char_vocabulary = set([char for sent in all_char_desc_list for char in str(sent).split('/')])
topic_vocabulary = set([char for sent in all_topics_list for char in str(sent).split('/')])
print "Size of the word vocabulary :", len(word_vocabulary)
print "Size of the char vocabulary :", len(char_vocabulary)
print "Number of topics : ", len(topic_vocabulary)
print "Preparing Vocabulary (complete)..."

###
# 3. vectorize data
###
print "\nVectorizing data..."
cv_word = CountVectorizer(vocabulary=word_vocabulary, token_pattern=u'(?u)\\b\\w+\\b')
cv_char= CountVectorizer(vocabulary=char_vocabulary, token_pattern=u'(?u)\\b\\w+\\b')
cv_topic = CountVectorizer(vocabulary=topic_vocabulary, token_pattern=u'(?u)\\b\\w+\\b')
print "Vectorizing data (complete)..."

###
# 4. combine data to 1 sparse matrix
###
print "\ncombining data..."
q_dict = dict()
u_dict = dict()
print "\tcompiling questions..."
for idx, entry in question_dataframe.iterrows():
    q_dict[entry['q_id']] = csr_matrix(hstack([cv_word.fit_transform([" ".join(entry['q_word_seq'].split('/'))]), [entry['q_tag']]]))
print "\tcompiling users..."
for idx, entry in user_dataframe.iterrows():
    u_dict[entry['u_id']] = csr_matrix(hstack([cv_word.fit_transform([" ".join(entry['e_desc_word_seq'].split('/'))]), cv_topic.fit_transform([" ".join(entry['e_expert_tags'].split('/'))])]))

X = list()
tempX = list()
y = list()
print "\tcompiling training info..."
for idx, entry in train_info_dataframe.iterrows():
    tempX.append(csr_matrix(hstack([q_dict[entry['q_id']], u_dict[entry['u_id']]])))
    y.append(entry['answered'])

X = csr_matrix(vstack(tempX))
print "combining data (complete)..."

###
# 5. train a Naive Bayes
###
print "\ntraining Naive Bayes..."
gnb = GaussianNB()
gnb_model = gnb.fit(csr_matrix(X).toarray(), y)
print "training Naive Bayes (complete)..."
