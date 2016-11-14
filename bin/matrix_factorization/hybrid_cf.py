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


NUM_THREADS = 20
NUM_COMPONENTS = 30
NUM_EPOCHS = 3
ITEM_ALPHA = 1e-6

question_path = "../../data/question_info.txt"
user_path = "../../data/user_info.txt"
invited_info_path = "../../data/invited_info_train.txt"


q_column_names = ['q_id', 'q_tag', 'q_word_seq', 'q_char_seq', 'q_no_upvotes', 'q_no_answers', 'q_no_quality_answers']
u_column_names = ['u_id','e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']
train_info_column_names = ['q_id','u_id','answered']

print 'reading data...'
question_dataframe = pd.read_csv(question_path, names=q_column_names, sep = '\t')
user_dataframe = pd.read_csv(user_path, names = u_column_names, sep = '\t')
train_info_dataframe = pd.read_csv(invited_info_path, names = train_info_column_names, sep = '\t')

print 'creating index map...'
q_index_map = {qid:i for i, qid in enumerate(question_dataframe['q_id'].tolist())}
u_index_map = {uid:i for i, uid in enumerate(user_dataframe['u_id'].tolist())}

X = []
y = []
for i, t in train_info_dataframe.iterrows():
    X.append([t['u_id'], t['q_id']])
    y.append(t['answered'])

X = np.array(X)
y = np.array(y)

#feature vector dict
print 'reading feature vectors...'
q_features = cPickle.load(open( "../../data/question_features.pkl", "rb" ))
u_features = cPickle.load(open( "../../data/user_features.pkl", "rb" ))


skf = StratifiedKFold(n_splits=9)
i = 0
for train_index, test_index in skf.split(X, y):
    print 'Cross-Validation run:{}, train_size:{}, test_size:{}'.format(i+1,len(train_index), len(test_index))
    i+=1
    Xtrain, ytrain = X[train_index], y[train_index]
    Xtest, ytest = X[test_index], y[test_index]
    train_matrix = np.zeros((len(user_dataframe), len(question_dataframe)))
    user_train_features = []
    question_train_features = []
    for i,X in enumerate(Xtrain):
        train_matrix[u_index_map[X[0]]][q_index_map[X[1]]] = int(ytrain[i])
        user_train_features.append(u_features[X[0]])
        question_train_features.append(q_features[X[1]])
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)
    model.fit(sparse.csr_matrix(train_matrix), epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

    Xtest_u_ids, Xtest_q_ids = zip(*Xtest)
    Xtest_u_ids = [u_index_map[id] for id in Xtest_u_ids]
    Xtest_q_ids = [q_index_map[id] for id in Xtest_q_ids]
    ypred = model.predict(Xtest_u_ids, Xtest_q_ids)

    train_auc = auc_score(model, train, num_threads=NUM_THREADS).mean()
    test_auc = roc_auc_score(ytest, ypred)
    print 'CF train AUC:{}'.format(train_auc)
    
    
    
