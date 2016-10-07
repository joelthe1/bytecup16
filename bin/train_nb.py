import numpy as np
import pandas as pd
import sys
import copy
import re
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
    f1 = cv_word.transform([re.sub("/"," ", entry['q_word_seq'])])
    f2 = cv_char.transform([re.sub("/"," ", entry['q_char_seq'])])
    f3 =  [entry['q_tag']]
    f4 = [entry['q_no_upvotes']]
    f5 = [entry['q_no_answers']]
    f6 = [entry['q_no_quality_answers']]
    q_dict[entry['q_id']] = csr_matrix(hstack([f1, f2, f3, f4, f5, f6]))

print "\tcompiling users..."
for idx, entry in user_dataframe.iterrows():
    u_column_names = ['u_id','e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']
    f1 = cv_word.transform([re.sub("/", " ", entry['e_desc_word_seq'])])
    f2 = cv_char.transform([re.sub("/", " ", entry['e_desc_char_seq'])])
    f3 = cv_topic.transform([re.sub("/", " ", entry['e_expert_tags'])])
    u_dict[entry['u_id']] = csr_matrix(hstack([f1, f2, f3]))

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
# 5. store the matrices
###

np_mat_file = open("np_mat.dat", 'w')
np.save(np_mat_file, X.toarray(), allow_pickle=True)
np_mat_file.close()

def save_sparse_csr(filename,array):
   np.savez(filename,data = array.data ,indices=array.indices,
            indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
   loader = np.load(filename)
   return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                        shape = loader['shape'])

save_sparse_csr("csr_mat.dat", X)

###
# 6. train a Naive Bayes
###
print "\ntraining Naive Bayes..."
gnb = GaussianNB()
gnb_model = gnb.fit(csr_matrix(X).toarray(), y)
print "training Naive Bayes (complete)..."
