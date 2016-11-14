import numpy as np
import pandas as pd
import sys
import copy
import cPickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, SpectralClustering


question_path = "../../data/question_info.txt"
user_path = "../../data/user_info.txt"
invited_info_path = "../../data/invited_info_train.txt"

q_column_names = ['q_id', 'q_tag', 'q_word_seq', 'q_char_seq', 'q_no_upvotes', 'q_no_answers', 'q_no_quality_answers']
u_column_names = ['u_id','e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']
train_info_column_names = ['q_id','u_id','answered']

question_dataframe = pd.read_csv(question_path, names=q_column_names, sep = '\t')
user_dataframe = pd.read_csv(user_path, names = u_column_names, sep = '\t')
train_info_dataframe = pd.read_csv(invited_info_path, names = train_info_column_names, sep = '\t')


all_word_desc_list = question_dataframe['q_word_seq'].tolist() + user_dataframe['e_desc_word_seq'].tolist()
all_char_desc_list = question_dataframe['q_char_seq'].tolist() + user_dataframe['e_desc_char_seq'].tolist()
all_topics_list = question_dataframe['q_tag'].tolist() + user_dataframe['e_expert_tags'].tolist()

word_vocabulary = set([word for sent in all_word_desc_list for word in str(sent).split('/')])
#possible inconsistency in the data ---- char seq np.nan found !
char_vocabulary = set([char for sent in all_char_desc_list for char in str(sent).split('/')])
topic_vocabulary = set([char for sent in all_topics_list for char in str(sent).split('/')])


print "Size of the word vocabulary :", len(word_vocabulary)
print "Size of the char vocabulary :", len(char_vocabulary)
print "Number of topics : ", len(topic_vocabulary)


cv_word = CountVectorizer(vocabulary=word_vocabulary, token_pattern=u'(?u)\\b\\w+\\b', ngram_range=(1,2))
cv_char= CountVectorizer(vocabulary=char_vocabulary, token_pattern=u'(?u)\\b\\w+\\b', ngram_range=(3,5))
cv_tag = CountVectorizer(vocabulary=topic_vocabulary, token_pattern=u'(?u)\\b\\w+\\b')

word_counts = cv_word.fit_transform(question_dataframe['q_word_seq'].tolist())
tf_word = TfidfTransformer(use_idf=True, norm='l2').fit(word_counts)
word_tfs = tf_word.transform(word_counts)
tsvd = TruncatedSVD(n_components=20, n_iter=7, random_state=42)
word_tfs = tsvd.fit_transform(word_tfs)
print "question word features", word_tfs.shape

char_counts = cv_char.fit_transform(question_dataframe['q_char_seq'].tolist())
tf_char = TfidfTransformer(use_idf=True, norm='l2').fit(char_counts)
char_tfs = tf_char.transform(char_counts)
tsvd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
char_tfs = tsvd.fit_transform(char_tfs)
print "question char features", char_tfs.shape

tag_counts = cv_tag.fit_transform([str(i) for i in question_dataframe['q_tag'].tolist()])
tf_tags = TfidfTransformer(use_idf=True, norm='l2').fit(tag_counts)
tag_tfs = tf_tags.transform(tag_counts)
tsvd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
tag_tfs = tsvd.fit_transform(tag_tfs)
print "question tag features", tag_tfs.shape

question_feature_matrix = Normalizer(copy=True).fit_transform(np.hstack([word_tfs, char_tfs, tag_tfs]))
print question_feature_matrix
print "question_feature_matrix original shape: ", question_feature_matrix.shape


kmeans = KMeans(n_clusters=7, random_state=0).fit(question_feature_matrix)
question_clusters1 = kmeans.labels_
sclustering = SpectralClustering(n_clusters=15, gamma=0.1).fit(question_feature_matrix)
question_clusters2 = sclustering.labels_


question_feature_map = {}
for i, q in question_dataframe.iterrows():
    question_feature_map[q['q_id']] = np.hstack([question_feature_matrix[i,:], q['q_no_upvotes'],
                                                 q['q_no_answers'], q['q_no_quality_answers'], question_clusters1, question_clusters2])

with open('question_features.pkl','wb') as fp:
    cPickle.dump(question_feature_map,fp)

question_feature_matrix = []
question_feature_map.clear()

# USER FEATURES

u_column_names = ['u_id','e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']    
word_counts = cv_word.fit_transform(user_dataframe['e_desc_word_seq'].tolist())
tf_word = TfidfTransformer(use_idf=True, norm='l2').fit(word_counts)
word_tfs = tf_word.transform(word_counts)
tsvd = TruncatedSVD(n_components=20, n_iter=7, random_state=42)
word_tfs = tsvd.fit_transform(word_tfs)
print "user word features", word_tfs.shape

char_counts = cv_char.fit_transform(user_dataframe['e_desc_char_seq'].tolist())
tf_char = TfidfTransformer(use_idf=True, norm='l2').fit(char_counts)
char_tfs = tf_char.transform(char_counts)
tsvd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
char_tfs = tsvd.fit_transform(char_tfs)
print "user char features", char_tfs.shape

tag_counts = cv_tag.fit_transform([str(i) for i in user_dataframe['e_expert_tags'].tolist()])
tf_tags = TfidfTransformer(use_idf=True, norm='l2').fit(tag_counts)
tag_tfs = tf_tags.transform(tag_counts)
tsvd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
tag_tfs = tsvd.fit_transform(tag_tfs)
print "user tag features", tag_tfs.shape

user_feature_matrix = Normalizer(copy=True).fit_transform(np.hstack([word_tfs, char_tfs, tag_tfs]))
print user_feature_matrix
print "user_feature_matrix original shape: ", user_feature_matrix.shape

kmeans = KMeans(n_clusters=7, random_state=0).fit(user_feature_matrix)
user_clusters1 = kmeans.labels_
sclustering = SpectralClustering(n_clusters=15, gamma=0.1).fit(user_feature_matrix)
user_clusters2 = sclustering.labels_

user_feature_map = {}
for i, u in user_dataframe.iterrows():
    user_feature_map[u['u_id']] = np.hstack([user_feature_matrix[i,:], user_clusters1, user_clusters2])

with open('user_features.pkl','wb') as fp:
    cPickle.dump(user_feature_map,fp)
