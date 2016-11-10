import numpy as np
import pandas as pd
import sys
import copy
import cPickle
from sklearn.decomposition import NMF , ProjectedGradientNMF
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

question_path = "../../data/question_info.txt"
user_path = "../../data/user_info.txt"
invited_info_path = "../../data/invited_info_train.txt"

q_column_names = ['q_id', 'q_tag', 'q_word_seq', 'q_char_seq', 'q_no_upvotes', 'q_no_answers', 'q_no_quality_answers']
u_column_names = ['u_id','e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']
train_info_column_names = ['q_id','u_id','answered']

question_dataframe = pd.read_csv(question_path, names=q_column_names, sep = '\t')
user_dataframe = pd.read_csv(user_path, names = u_column_names, sep = '\t')
train_info_dataframe = pd.read_csv(invited_info_path, names = train_info_column_names, sep = '\t')

print "initializing..."
train_map = {}
q_index_map = {qid:i for i, qid in enumerate(question_dataframe['q_id'].tolist())}
u_index_map = {uid:i for i, uid in enumerate(user_dataframe['u_id'].tolist())}
X = []
y = []
for i, t in train_info_dataframe.iterrows():
    train_map.setdefault(t['q_id'], {})[t['u_id']]=float(t['answered'])
    X.append([t['q_id'], t['u_id']])
    y.append(t['answered'])

X = np.array(X)
y = np.array(y)
best_mse = np.inf
skf = cross_validation.StratifiedKFold(y, n_folds=9)

for nc in [50, 100, 120, 125, 130, 140, 150]:
    mse = []
    for train_index, test_index in skf:
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        feature_matrix = np.zeros((len(question_dataframe['q_id'].tolist()),
                                   len(user_dataframe['u_id'].tolist())))
        # print "creating the feature matrix..."
        # print "train size : {} , test size : {}".format(len(X_train), len(X_test))
        for i in range(len(X_train)):
            iq = q_index_map[X_train[i][0]]
            iu = u_index_map[X_train[i][1]]
            if X_train[i][0] in train_map:
                feature_matrix[iq][iu] = train_map[X_train[i][0]].get(X_train[i][1], 0.0)
        # print "running non-negative matrix factorization..."
        nmf = NMF(solver='cd', alpha=0.2, tol=1e-4, max_iter=300, n_components = nc)
        W = nmf.fit_transform(feature_matrix)
        # print "finished nmf fit_transform.."
        H = nmf.components_
        # print "W : {}, H : {}".format(W.shape, H.shape)
        # print  "calculating the resulting matrix..."
        nR = np.dot(W,H)
        y_pred = []
        y_true = []
        for i in range(len(X_test)):
            iq = q_index_map[X_test[i][0]]
            iu = u_index_map[X_test[i][1]]
            y_true.append(train_map[X_test[i][0]][X_test[i][1]])
            y_pred.append(nR[iq][iu])
        mse.append(mean_squared_error(y_true, y_pred))
    if best_mse > np.mean(mse):
        best_mse = min(np.mean(mse), best_mse)
        best_nc = nc
    print "avg_mse:{}, n_components:{} best_mse_so_far:{} best_nc_so_far:{}".format(np.mean(mse), nc, best_mse, best_nc)




