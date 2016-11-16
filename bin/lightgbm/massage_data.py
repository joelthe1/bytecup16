import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
import pickle
import sys
sys.path.insert(0, '~/zmod/bin/feature_engineering')
import lsa

train_info_dataframe = pd.read_csv("../../data/invited_info_train.txt", names = ['q_id','u_id','answered'], sep = '\t')
validate_ids_dataframe = pd.read_pickle("../../data/validate_nolabel.pkl")

print "\n Loading data mapings"
q_dict, u_dict = lsa.run()
print "Loading data mapings(complete)"

print "\ncombining both user and question data..."

for i,data_source in enumerate([train_info_dataframe, validate_ids_dataframe]):
    if i==0:
        continue
    res_bclass = []
    res_rank = []
    res_reg = []

    for idx, entry in data_source.iterrows():
        line = hstack([q_dict[entry['q_id']], u_dict[entry['u_id']]]).toarray()

        # Binary classification
        x_bclass = map(lambda x: '{:.18f}'.format(x), line[0])
        if i == 0:
            res_bclass.append(str(entry['answered']) + '\t' + '\t'.join(x_bclass))  #training data
        else:
            res_bclass.append('1\t' + '\t'.join(x_bclass))                          #test data
        
        # Regression
        #x_reg = map(lambda x: '{:.18f}'.format(x), line[0])
        #if i == 0:
        #    res_reg.append('1\t' + '\t'.join(x_reg))
        #else:
        #    res_reg.append(str(entry['answered']) + '\t' + '\t'.join(x_reg))

        # Rank
        #x = map(lambda (i, x): '{}:{:.18f}'.format((i+1), x) if x!=0 else 0, enumerate(line[0]))
        #x = filter(lambda x: type(x) == str, x)
        #t = '1\t' + '\t'.join(x)
        #    t = str(entry['answered']) + '\t' + '\t'.join(x)
        # res_rank.append(t)

    if i == 0:
        wfile = open('bien_bclassifier.train', 'w')
        wfile.write('\n'.join(res_bclass))
        wfile.close()
    else:
        wfile = open('bien_bclassifier.test', 'w')
        wfile.write('\n'.join(res_bclass))
        wfile.close
'''
wfile = open('bien_rank.test', 'w')
wfile.write('\n'.join(res_rank))
wfile.close()

wfile = open('bien_regression.test', 'w')
wfile.write('\n'.join(res_reg))
wfile.close()
'''
#X = csr_matrix(vstack(tempX))
#save_sparse_csr("../../data/csr_mat_train_lsa.dat", X)

print "combining data and save(complete)"
print "\nLoading pca data(complete)"
