import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import auc_score
from collections import Counter
sys.path.append('../')
from feature_engineering.load_data import loadData


NUM_THREADS = 24

data = loadData('../../data')
X = data.train.as_matrix(['q_id', 'u_id'])
y = np.array(data.train['answered'].tolist())

q_index_map = {q['q_id']:i  for i,q in data.questions.iterrows()}
u_index_map = {u['u_id']:i  for i,u in data.users.iterrows()}

def run(components, epochs, loss):
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
        model = LightFM(loss=loss,
                        max_sampled=20,
                        no_components=components)
        model.fit(train_matrix,
                  epochs=epochs, num_threads=NUM_THREADS)
        
        train_auc = auc_score(model, train_matrix,
                              num_threads=NUM_THREADS).mean()
        test_auc = auc_score(model, test_matrix,
                             num_threads=NUM_THREADS).mean()
        auc.append(test_auc)
        
    print '----------->cross-validation mean AUC on test: {} components:{} epochs:{} loss:{}'.format(np.mean(auc), components, epochs, loss)
    return np.mean(auc)

def generate_validation(filename='temp.csv'):
    X = data.train.as_matrix(['q_id', 'u_id'])
    y = np.array(data.train['answered'].tolist())
    train_row = []
    train_column = []
    train_label = []
    for i,XV in enumerate(X):
        train_row.append(u_index_map[XV[1]])
        train_column.append(q_index_map[XV[0]])
        train_label.append(int(y[i]))
    train_matrix = sparse.csr_matrix((train_label,(train_row, train_column)),
                                     shape=(len(data.users), len(data.questions)),
                                     dtype=np.int32)
    model = LightFM(loss='warp-kos',
                    max_sampled=20,
                    no_components=55)
    model.fit(train_matrix,
              epochs=68, num_threads=NUM_THREADS)
    train_auc = auc_score(model, train_matrix,
                          num_threads=NUM_THREADS).mean()
    print 'train_auc for validation generation:', train_auc
    Xtest_q_ids, Xtest_u_ids = zip(*data.validation.as_matrix(['q_id', 'u_id']))
    Xtest_u_ids = np.array([u_index_map[id] for id in Xtest_u_ids])
    Xtest_q_ids = np.array([q_index_map[id] for id in Xtest_q_ids])
    ypred = model.predict(Xtest_u_ids, Xtest_q_ids,
                          num_threads = NUM_THREADS)
    with open(filename, 'w') as f:
        f.write('qid,uid,label\n')
        assert len(ypred) == len(data.validation)
        for i, v in data.validation.iterrows():
            f.write('{},{},{}\n'.format(v['q_id'],v['u_id'],ypred[i]))
    return
        
    
    

if __name__=='__main__':
    # best_config = None
    # best_auc = -np.inf
    # best_loss = None
    # loss_functions = ['warp-kos']
    # for loss in loss_functions:
    #     for components in [40, 45, 50, 55]:
    #         for epochs in [65, 67, 68, 70]:
    #             auc = run(components, epochs, loss)
    #             if auc>best_auc:
    #                 best_config = (components, epochs, loss)
    #                 best_auc = auc
    #                 best_loss = loss
    # print "best AUC:{} CONFIG:{} LOSS:{}".format(best_auc, best_config, best_loss)
    generate_validation()
