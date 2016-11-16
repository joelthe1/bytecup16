import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
import pickle

train_info_dataframe = pd.read_csv("../../data/invited_info_train.txt", names = ['q_id','u_id','answered'], sep = '\t')
validate_ids_dataframe = pd.read_pickle("../../data/validate_nolabel.pkl")

print "\nLoading pca data..."
print "Loading user data..."
u_dict = pickle.load(open("user_features.pkl",'r'))
print "Loading user data(complete)"
print "Loading question data..."
q_dict = pickle.load(open("question_features.pkl",'r'))
print "Loading question data(complete)"

print "\ncombining both user and question data..."

feat_ids = range(1,844,1)

res_rank = []
res_reg = []
for idx, entry in validate_ids_dataframe.iterrows():
    line = hstack([q_dict[entry['q_id']], u_dict[entry['u_id']]]).toarray()

    # Regression
    x_reg = map(lambda x: '{:.18f}'.format(x), line[0])
    res_reg.append('1\t' + '\t'.join(x_reg))
#    res_reg.append(str(entry['answered']) + '\t' + '\t'.join(x_reg))

    # Rank
    x = map(lambda (i, x): '{}:{:.18f}'.format((i+1), x) if x!=0 else 0, enumerate(line[0]))
    x = filter(lambda x: type(x) == str, x)
    t = '1\t' + '\t'.join(x)
#    t = str(entry['answered']) + '\t' + '\t'.join(x)
    res_rank.append(t)

wfile = open('bien_rank.test', 'w')
wfile.write('\n'.join(res_rank))
wfile.close()

wfile = open('bien_regression.test', 'w')
wfile.write('\n'.join(res_reg))
wfile.close()
#X = csr_matrix(vstack(tempX))
#save_sparse_csr("../../data/csr_mat_train_lsa.dat", X)

print "combining data and save(complete)"
print "\nLoading pca data(complete)"
