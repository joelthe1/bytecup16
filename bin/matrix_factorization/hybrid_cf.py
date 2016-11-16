import numpy as np
import pandas as pd
import sys
import copy
import cPickle
from sklearn.decomposition import NMF , ProjectedGradientNMF
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from lightfm.datasets import fetch_stackexchange
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import auc_score
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, SpectralClustering

NUM_THREADS = 24
NUM_COMPONENTS = 35
NUM_EPOCHS = 45
ITEM_ALPHA = 1e-4
USER_ALPHA = 1e-4
print 'epochs:{}, components:{}, item-alpha:{}, user-alpha:{}'.format(NUM_EPOCHS, NUM_COMPONENTS, ITEM_ALPHA, USER_ALPHA)
question_path = "../../data/question_info.txt"
user_path = "../../data/user_info.txt"
invited_info_path = "../../data/invited_info_train.txt"
validation_nolabel_path = '../../data/validate_nolabel.txt'

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
char_vocabulary = set([char for sent in all_char_desc_list for char in str(sent).split('/')])
topic_vocabulary = set([char for sent in all_topics_list for char in str(sent).split('/')])

cv_word = CountVectorizer(vocabulary=word_vocabulary, token_pattern=u'(?u)\\b\\w+\\b', ngram_range=(1,2))
cv_char= CountVectorizer(vocabulary=char_vocabulary, token_pattern=u'(?u)\\b\\w+\\b', ngram_range=(2,5))
cv_tag = CountVectorizer(vocabulary=topic_vocabulary, token_pattern=u'(?u)\\b\\w+\\b')

word_counts = cv_word.fit_transform(question_dataframe['q_word_seq'].tolist())
tf_word = TfidfTransformer(use_idf=True, norm='l2').fit(word_counts)
word_tfs = tf_word.transform(word_counts)
tsvd = TruncatedSVD(n_components=150, n_iter=15, random_state=42)
word_tfs = tsvd.fit_transform(word_tfs)


char_counts = cv_char.fit_transform(question_dataframe['q_char_seq'].tolist())
tf_char = TfidfTransformer(use_idf=True, norm='l2').fit(char_counts)
char_tfs = tf_char.transform(char_counts)
tsvd = TruncatedSVD(n_components=100, n_iter=15, random_state=42)
char_tfs = tsvd.fit_transform(char_tfs)


tag_counts = cv_tag.fit_transform([str(i) for i in question_dataframe['q_tag'].tolist()])
tf_tags = TfidfTransformer(use_idf=True, norm='l2').fit(tag_counts)
tag_tfs = tf_tags.transform(tag_counts)
tsvd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
tag_tfs = tsvd.fit_transform(tag_tfs)

question_feature_matrix = Normalizer(copy=True).fit_transform(np.hstack([word_tfs,
                                                                         question_dataframe.as_matrix(['q_no_upvotes','q_no_answers','q_no_quality_answers']),
                                                                         tag_tfs]))

question_feature_map = {}
question_features = []
for i, q in question_dataframe.iterrows():
    question_features.append(np.hstack([question_feature_matrix[i,:]]))

# USER FEATURES

u_column_names = ['u_id','e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']    
word_counts = cv_word.fit_transform(user_dataframe['e_desc_word_seq'].tolist())
tf_word = TfidfTransformer(use_idf=True, norm='l2').fit(word_counts)
word_tfs = tf_word.transform(word_counts)
tsvd = TruncatedSVD(n_components=150, n_iter=15, random_state=42)
word_tfs = tsvd.fit_transform(word_tfs)


char_counts = cv_char.fit_transform(user_dataframe['e_desc_char_seq'].tolist())
tf_char = TfidfTransformer(use_idf=True, norm='l2').fit(char_counts)
char_tfs = tf_char.transform(char_counts)
tsvd = TruncatedSVD(n_components=100, n_iter=15, random_state=42)
char_tfs = tsvd.fit_transform(char_tfs)


tag_counts = cv_tag.fit_transform([str(i) for i in user_dataframe['e_expert_tags'].tolist()])
tf_tags = TfidfTransformer(use_idf=True, norm='l2').fit(tag_counts)
tag_tfs = tf_tags.transform(tag_counts)
tsvd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
tag_tfs = tsvd.fit_transform(tag_tfs)

user_feature_matrix = Normalizer(copy=True).fit_transform(np.hstack([word_tfs, tag_tfs]))

user_feature_map = {}
user_features = []
for i, u in user_dataframe.iterrows():
    user_features.append(np.hstack([user_feature_matrix[i,:]]))


q_index_map = {qid:i for i, qid in enumerate(question_dataframe['q_id'].tolist())}
u_index_map = {uid:i for i, uid in enumerate(user_dataframe['u_id'].tolist())}

X = []
y = []
for i, t in train_info_dataframe.iterrows():
    X.append([t['u_id'], t['q_id']])
    y.append(t['answered'])

X = np.array(X)
y = np.array(y)
skf = StratifiedKFold(n_splits=5)
ci = 0
auc = []
for train_index, test_index in skf.split(X, y):
    ci+=1
    Xtrain, ytrain = X[train_index], y[train_index]
    Xtest, ytest = X[test_index], y[test_index]
    train_matrix = np.zeros((len(user_dataframe), len(question_dataframe)), dtype=float)
    test_matrix = np.zeros((len(user_dataframe), len(question_dataframe)), dtype=float)
    user_train_features = []
    user_test_features = []
    question_train_features = []
    question_test_features = []
    for i,XV in enumerate(Xtrain):
        train_matrix[u_index_map[XV[0]]][q_index_map[XV[1]]] = 1.0 if int(ytrain[i]) is 1 else -1.0
    for i,XV in enumerate(Xtest):
        test_matrix[u_index_map[XV[0]]][q_index_map[XV[1]]] = 1.0 if int(ytest[i]) is 1 else -1.0
        
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    user_alpha=USER_ALPHA,
                    no_components=NUM_COMPONENTS)
    model.fit(sparse.csr_matrix(train_matrix, dtype=np.float32),
              user_features=sparse.csr_matrix(user_features, dtype=np.float32),
              item_features=sparse.csr_matrix(question_features, dtype=np.float32),
              epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

    Xtest_u_ids, Xtest_q_ids = zip(*Xtest)
    Xtest_u_ids = np.array([u_index_map[id] for id in Xtest_u_ids])
    Xtest_q_ids = np.array([q_index_map[id] for id in Xtest_q_ids])
    ypred = model.predict(Xtest_u_ids, Xtest_q_ids,
                          user_features = sparse.csr_matrix(user_features, dtype=np.float32),
                          item_features = sparse.csr_matrix(question_features, dtype=np.float32),
                          num_threads = NUM_THREADS)
                      
    train_auc = auc_score(model, sparse.csr_matrix(train_matrix, dtype=np.float32),
                          user_features = sparse.csr_matrix(user_features, dtype=np.float32),
                          item_features = sparse.csr_matrix(question_features, dtype=np.float32),
                          num_threads=NUM_THREADS).mean()
    test_auc = auc_score(model, sparse.csr_matrix(test_matrix, dtype=np.float32),
                          user_features = sparse.csr_matrix(user_features, dtype=np.float32),
                          item_features = sparse.csr_matrix(question_features, dtype=np.float32),
                          num_threads=NUM_THREADS).mean()
    
    #test_auc = roc_auc_score(ytest, ypred)
    auc.append(test_auc)
    print 'CF train AUC:{} test AUC:{}'.format(train_auc, test_auc)

print 'cross-validation mean AUC on test: {}'.format(np.mean(auc))

with open(validation_nolabel_path, 'r') as f:
    lines = f.readlines()
    Xtest = []
    train_matrix = np.zeros((len(user_dataframe), len(question_dataframe)), dtype=float)
    for line in lines[1:]:
        line = line.strip()
        qid = line.split(',')[0]
        uid = line.split(',')[1]
        Xtest.append([uid, qid])
    for i,XV in enumerate(X):
        train_matrix[u_index_map[XV[0]]][q_index_map[XV[1]]] = 1.0 if int(y[i]) is 1 else -1.0
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    user_alpha=USER_ALPHA,
                    no_components=NUM_COMPONENTS)
    model.fit(sparse.csr_matrix(train_matrix, dtype=np.float32),
              user_features=sparse.csr_matrix(user_features, dtype=np.float32),
              item_features=sparse.csr_matrix(question_features, dtype=np.float32),
              epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
    train_auc = auc_score(model, sparse.csr_matrix(train_matrix, dtype=np.float32),
                          user_features = sparse.csr_matrix(user_features, dtype=np.float32),
                          item_features = sparse.csr_matrix(question_features, dtype=np.float32),
                          num_threads=NUM_THREADS).mean()
    print 'train auc:{}'.format(train_auc)
    Xtest_u_ids, Xtest_q_ids = zip(*Xtest)
    Xtest_u_ids = np.array([u_index_map[id] for id in Xtest_u_ids])
    Xtest_q_ids = np.array([q_index_map[id] for id in Xtest_q_ids])
    ypred = model.predict(Xtest_u_ids, Xtest_q_ids,
                          user_features = sparse.csr_matrix(user_features, dtype=np.float32),
                          item_features = sparse.csr_matrix(question_features, dtype=np.float32),
                          num_threads = NUM_THREADS)

with open('temp.csv', 'w') as f:
    final_lines = ['qid,uid,label\n']
    for i,line in enumerate(lines[1:]):
        line = line.strip()
        qid = line.split(',')[0]
        uid = line.split(',')[1]
        final_lines.append('{},{},{}\n'.format(qid, uid, ypred[i]))
    for line in final_lines:
        f.write(line)
    
        
        
    
        
    
