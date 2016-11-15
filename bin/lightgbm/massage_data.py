import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
import pickle

train_info_dataframe = pd.read_csv("../../data/invited_info_train.txt", names = ['q_id','u_id','answered'], sep = '\t')

print "\nLoading pca data..."
print "Loading user data..."
u_dict = pickle.load(open("user_features.pkl",'r'))
print "Loading user data(complete)"
print "Loading question data..."
q_dict = pickle.load(open("question_features.pkl",'r'))
print "Loading question data(complete)"

X = list()
tempX = list()
y = list()
print "\ncombining both user and question data..."

feat_ids = range(1,844,1)

for idx, entry in train_info_dataframe.iterrows():
    line = hstack([q_dict[entry['q_id']], u_dict[entry['u_id']]])
    y.append(entry['answered'])
    x = map(lambda (i, x): '{}:{:.18f}'.format((i+1), x), enumerate(line.toarray()[0]))
    print x
#    print dict(zip(feat_ids, x))
    break

#X = csr_matrix(vstack(tempX))
#save_sparse_csr("../../data/csr_mat_train_lsa.dat", X)

print "combining data and save(complete)"
print "\nLoading pca data(complete)"
