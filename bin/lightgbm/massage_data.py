import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
import pickle
import sys
sys.path.insert(0, '~/zmod/bin/feature_engineering')
import lsa

train_info_dataframe = pd.read_csv("../../data/invited_info_train.txt", names = ['q_id','u_id','answered'], sep = '\t')
validate_ids_dataframe = pd.read_pickle("../../data/validate_nolabel.pkl")
test_ids_dataframe = pd.read_pickle("../../data/test_nolabel.pkl")

print "\n Loading data mapings"
n_comp = [500, 500, 20, 700, 700, 30]
q_dict, u_dict = lsa.run(n_comp)
print "Loading data mapings(complete)"

print "\ncombining both user and question data..."

for i,data_source in enumerate([train_info_dataframe, validate_ids_dataframe, test_ids_dataframe]):
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
            res_bclass.append('\t'.join(x_bclass))                          #test data
            
        
        # Regression
        #x_reg = map(lambda x: '{:.18f}'.format(x), line[0])
        #if i == 0:
        #    res_reg.append('\t'.join(x_reg))
        #else:
        #    res_reg.append(str(entry['answered']) + '\t' + '\t'.join(x_reg))

        # Rank
        #x = map(lambda (i, x): '{}:{:.18f}'.format((i+1), x) if x!=0 else 0, enumerate(line[0]))
        #x = filter(lambda x: type(x) == str, x)
        #t = '\t'.join(x)
        #    t = str(entry['answered']) + '\t'.join(x)
        # res_rank.append(t)

    if i == 0:
        print "\nsplitting data into training and validation set..."
        #split training data into training and validation sets
        n = len(res_bclass)
        #do a 80-20 split
        m = int(0.2 * n)
        v_idx = np.random.randint(n, size=m)
        vset = list()
        tset = list()
        for i in range(n):
            if i in v_idx:
                vset.append(res_bclass[i])
            else:
                tset.append(res_bclass[i])

        print "splitting data into training and validation set(comleted)..."
                
        print "\nsaving training set to file..."
        wfile = open('bien_bclassifier.train', 'w')
        wfile.write('\n'.join(tset))
        wfile.close()
        print "saving training set to file(complete)..."

        print "\nsaving validation set to file..."
        wfile = open('bien_bclassifier.val', 'w')
        wfile.write('\n'.join(vset))
        wfile.close()
        print "saving validation set to file(complete)..."
    elif i == 1:
        print "\nsaving test set to file..."
        wfile = open('bien_bclassifier.test', 'w')
        wfile.write('\n'.join(res_bclass))
        wfile.close
        print "saving test set to file(complete)..."
    elif i == 2:
        print "\nsaving final test file..."
        wfile = open('bien_bclassifier.final', 'w')
        wfile.write('\n'.join(res_bclass))
        wfile.close
        print "saving final test file(complete)..."

print "combining data and save(complete)"
print "\nLoading pca data(complete)"
