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
validate_nolabel_path = "../../data/validate_nolabel.txt"

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
feature_matrix = np.zeros((len(question_dataframe['q_id'].tolist()),
                           len(user_dataframe['u_id'].tolist())))
for i, t in train_info_dataframe.iterrows():
    train_map.setdefault(t['q_id'], {})[t['u_id']]=float(t['answered'])
    X.append([t['q_id'], t['u_id']])
    y.append(t['answered'])

X = np.array(X)
y = np.array(y)
for i in range(len(X)):
    iq = q_index_map[X[i][0]]
    iu = u_index_map[X[i][1]]
    if X[i][0] in train_map:
        feature_matrix[iq][iu] = train_map[X[i][0]].get(X[i][1], 0.0)
nmf = NMF(solver='cd', alpha=0.2, tol=1e-4, max_iter=300, n_components = 100)
W = nmf.fit_transform(feature_matrix)
# print "finished nmf fit_transform.."
H = nmf.components_
# print "W : {}, H : {}".format(W.shape, H.shape)
# print  "calculating the resulting matrix..."
nR = np.dot(W,H)

final_lines = ['qid,uid,label\n']
with open(validate_nolabel_path,'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        qid = line.split(',')[0]
        uid = line.split(',')[1]
        if qid in q_index_map and uid in u_index_map:
            pred = nR[q_index_map[qid]][u_index_map[uid]]
        final_lines.append('{},{},{}\n'.format(qid, uid, pred))
with open('temp.csv','w') as f:
    for line in final_lines:
        f.write(line)
        
