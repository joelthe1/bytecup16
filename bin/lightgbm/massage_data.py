import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.model_selection import StratifiedKFold
import pickle
import sys
sys.path.insert(0, '../feature_engineering')
from load_data import loadData

data = loadData('../../data')
q_mat, u_mat, y = data.training_features()

train_data = np.hstack([q_mat, u_mat])

skf = StratifiedKFold(n_splits=5, random_state=2016)
count = 0
for train_index, test_index in skf.split(X, y):
    count += 1
    Xtrain, ytrain = X[train_index], y[train_index]
    Xtest, ytest = X[test_index], y[test_index]
    print Xtrain.shape, ytrain.shape

    train_data = []
    for idx, data_row in enumerate(Xtrain):
        Xtrain_str = map(lambda x: '{:.18f}'.format(x), data_row)
        train_data.append(str(ytrain[idx]) + '\t' + '\t'.join(Xtrain_str))

    wfile = open('lgbm.train{}'.format(count), 'w')
    wfile.write('\n'.join(train_data))
    wfile.close()

    dev_data = []
    for idx, data_row in enumerate(Xtest):
        Xtest_str = map(lambda x: '{:.18f}'.format(x), data_row)
        dev_data.append(str(ytest[idx]) + '\t' + '\t'.join(Xtest_str))

    wfile = open('lgbm.dev{}'.format(count), 'w')
    wfile.write('\n'.join(dev_data))
    wfile.close()


        

# print "\n Loading data mapings"
# n_comp = [500, 500, 20, 700, 700, 30]
# q_dict, u_dict = lsa.run(n_comp)
# print "Loading data mapings(complete)"

# print "\ncombining both user and question data..."

# for i,data_source in enumerate([train_info_dataframe, validate_ids_dataframe, test_ids_dataframe]):
#     res_bclass = []
#     res_rank = []
#     res_reg = []
#     y = []
#     X = []

#     # for idx, entry in data_source.iterrows():
#     #     if idx == 100:
#     #         break
#     #     line = np.hstack([q_dict[entry['q_id']], u_dict[entry['u_id']]]).toarray()

#     #     # Binary classification
#     #     x_bclass = map(lambda x: '{:.18f}'.format(x), line[0])
#     #     if i == 0:
#     #         y.append(str(entry['answered']))
#     #         X.append('\t'.join(x_bclass))  #training data
#     #     else:
#             # res_bclass.append('\t'.join(x_bclass))                          #test data
            

#         # Regression
#         #x_reg = map(lambda x: '{:.18f}'.format(x), line[0])
#         #if i == 0:
#         #    res_reg.append('\t'.join(x_reg))
#         #else:
#         #    res_reg.append(str(entry['answered']) + '\t' + '\t'.join(x_reg))

#         # Rank
#         #x = map(lambda (i, x): '{}:{:.18f}'.format((i+1), x) if x!=0 else 0, enumerate(line[0]))
#         #x = filter(lambda x: type(x) == str, x)
#         #t = '\t'.join(x)
#         #    t = str(entry['answered']) + '\t'.join(x)
#         # res_rank.append(t)

#     if i == 0:
#         print "\nsplitting data into training and validation set..."
#         #split training data into training and validation sets
#         n = len(res_bclass)
#         #do a 80-20 split
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)
#         print X_train[0], y_train[0], X_test[0], y_test[0]
#         exit()
#         m = int(0.2 * n)
        
#         vset = list()
#         tset = list()
#         for i in range(n):
#             if i in v_idx:
#                 vset.append(res_bclass[i])
#             else:
#                 tset.append(res_bclass[i])

#         print "splitting data into training and validation set(comleted)..."
                
#         print "\nsaving training set to file..."
#         wfile = open('bien_bclassifier.train', 'w')
#         wfile.write('\n'.join(tset))
#         wfile.close()
#         print "saving training set to file(complete)..."

#         print "\nsaving validation set to file..."
#         wfile = open('bien_bclassifier.val', 'w')
#         wfile.write('\n'.join(vset))
#         wfile.close()
#         print "saving validation set to file(complete)..."
#     elif i == 1:
#         print "\nsaving test set to file..."
#         wfile = open('bien_bclassifier.test', 'w')
#         wfile.write('\n'.join(res_bclass))
#         wfile.close
#         print "saving test set to file(complete)..."
#     elif i == 2:
#         print "\nsaving final test file..."
#         wfile = open('bien_bclassifier.final', 'w')
#         wfile.write('\n'.join(res_bclass))
#         wfile.close
#         print "saving final test file(complete)..."

# print "combining data and save(complete)"
# print "\nLoading pca data(complete)"
