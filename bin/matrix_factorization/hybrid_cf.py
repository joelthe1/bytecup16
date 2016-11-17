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
sys.path.append('../')
from feature_engineering.load_data import loadData


NUM_THREADS = 24
NUM_COMPONENTS = 75
NUM_EPOCHS = 50
ITEM_ALPHA = 1e-4
USER_ALPHA = 1e-4

data = loadData('../../data')
X, y = data.raw_training_data()
question_features, user_features = data.nmf_array(components=100)
q_index_map = {q['q_id']:i  for i,q in data.questions.iterrows()}
u_index_map = {u['u_id']:i  for i,u in data.users.iterrows()}

skf = StratifiedKFold(n_splits=5, random_state=2016)
ci = 0
auc = []
for train_index, test_index in skf.split(X, y):
    ci+=1

    Xtrain, ytrain = X[train_index], y[train_index]
    Xtest, ytest = X[test_index], y[test_index]

    train_row = []
    train_column = []
    train_label = []
    for i,XV in enumerate(Xtrain):
        train_row.append(u_index_map[XV[1]])
        train_column.append(q_index_map[XV[0]])
        train_label.append(int(ytrain[i]))
    train_matrix = sparse.csr_matrix((train_label,(train_row, train_column)),
                                     shape=(len(data.users), len(data.questions)),
                                     dtype=np.int32)
    test_row = []
    test_column = []
    test_label = []
    for i,XV in enumerate(Xtest):
        test_row.append(u_index_map[XV[1]])
        test_column.append(q_index_map[XV[0]])
        test_label.append(int(ytest[i]))
    test_matrix = sparse.csr_matrix((test_label, (test_row, test_column)),
                                    shape=(len(data.users), len(data.questions)),
                                     dtype=np.int32)
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    user_alpha=USER_ALPHA,
                    no_components=NUM_COMPONENTS)
    model.fit(train_matrix,
              user_features=sparse.csr_matrix(user_features, dtype=np.float32),
              item_features=sparse.csr_matrix(question_features, dtype=np.float32),
              epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

    Xtest_q_ids, Xtest_u_ids = zip(*Xtest)
    Xtest_u_ids = np.array([u_index_map[id] for id in Xtest_u_ids])
    Xtest_q_ids = np.array([q_index_map[id] for id in Xtest_q_ids])
    ypred = model.predict(Xtest_u_ids, Xtest_q_ids,
                          user_features = sparse.csr_matrix(user_features, dtype=np.float32),
                          item_features = sparse.csr_matrix(question_features, dtype=np.float32),
                          num_threads = NUM_THREADS)
                      
    train_auc = auc_score(model, train_matrix,
                          user_features = sparse.csr_matrix(user_features, dtype=np.float32),
                          item_features = sparse.csr_matrix(question_features, dtype=np.float32),
                          num_threads=NUM_THREADS).mean()
    test_auc = auc_score(model, test_matrix,
                          user_features = sparse.csr_matrix(user_features, dtype=np.float32),
                          item_features = sparse.csr_matrix(question_features, dtype=np.float32),
                          num_threads=NUM_THREADS).mean()
    
    #test_auc = roc_auc_score(ytest, ypred)
    auc.append(test_auc)
    print 'CF train AUC:{} test AUC:{}'.format(train_auc, test_auc)

print 'cross-validation mean AUC on test: {}'.format(np.mean(auc))


print 'Generating temp.csv on validation_nolabel.txt'
validate_nolabel_path = "../../data/validate_nolabel.txt"
with open(validate_nolabel_path, 'r') as f:
    lines = f.readlines()
    Xtest = []
    row = []
    column = []
    label = []
    for line in lines[1:]:
        line = line.strip()
        qid = line.split(',')[0]
        uid = line.split(',')[1]
        Xtest.append([qid, uid])
    for i,XV in enumerate(X):
        row.append(u_index_map[XV[1]])
        column.append(q_index_map[XV[0]])
        label.append(int(y[i]))
    train_matrix = sparse.csr_matrix((label,(row, column)),
                                     shape=(len(data.users), len(data.questions)),
                                     dtype=np.int32)
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    user_alpha=USER_ALPHA,
                    no_components=NUM_COMPONENTS)
    model.fit(train_matrix,
              user_features=sparse.csr_matrix(user_features, dtype=np.float32),
              item_features=sparse.csr_matrix(question_features, dtype=np.float32),
              epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
    train_auc = auc_score(model, train_matrix,
                          user_features = sparse.csr_matrix(user_features, dtype=np.float32),
                          item_features = sparse.csr_matrix(question_features, dtype=np.float32),
                          num_threads=NUM_THREADS).mean()
    print 'train auc:{}'.format(train_auc)
    Xtest_q_ids, Xtest_u_ids = zip(*Xtest)
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
    
        
        
    
        
    
