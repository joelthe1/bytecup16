import pandas as pd
import numpy as np
import xgboost
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix, hstack, vstack
import pickle
import sys
sys.path.insert(0, '../feature_engineering')
from load_data import loadData
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import errno
import datetime


class xgboost_wrapper:
    def __init__(self):
        data = loadData('../../data')
        self.X, self.y, self.X_valid, self.X_test = data.dataset()

        self.X_valid = None
        self.X_test = None

        # self.validate_ids_dataframe = data.validation
        # self.test_ids_df = data.test
        
        self.model = None;
        
        self.params = {'max_depth':9,
                       'n_estimators':180,
                       'learning_rate':0.02,
                       'silent':True,
                       'objective':'binary:logistic',
                       'gamma':0,
                       'min_child_weight':7,
                       'max_delta_step':6,
                       'subsample':0.9,
                       'reg_lambda':3,
                       'reg_alpha':1,
                       'scale_pos_weight':1,
                       'colsample_bytree':0.9}

    def make_dir(self, path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def genstats(self, info, iteration, dirname):
        self.make_dir(dirname)
        wfile = open('{}/stats_cv_{}.out'.format(dirname, iteration), 'w')
	for vset in info:
            for loss in info[vset]:
                y =info[vset][loss]
                plt.plot(y)
                plt.xticks(np.arange(0, len(y)+1, 10.0))

                max_val = float('-inf')
                for i,x in enumerate(y):
                    if max_val < x:
                        max_val = x
                        idx = i
                wfile.write('max is '+ str(max_val) + ' at ' + str(idx))
        wfile.write('\n')
        plt.cfl()
        plt.savefig('{}/plot_cv_{}.png'.format(dirname, iteration), bbox_inches='tight')
        wfile.close()

    def cross_validate(self):
        time_now = '%s' % datetime.datetime.now()
        dtrain = xgboost.DMatrix(self.X, label=self.y)
        skf = StratifiedKFold(n_splits=5, random_state=2016)
        i = -1
        for train_index, test_index in skf.split(self.X, self.y):
            i += 1
            # if i == 0:
            #     continue
            Xtrain, ytrain = self.X[train_index], self.y[train_index]
            Xtest, ytest = self.X[test_index], self.y[test_index]
            print Xtrain.shape, ytrain.shape

            self.model = xgboost.XGBClassifier(**self.params).fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)], eval_metric='auc', verbose=True, early_stopping_rounds=10)
            print self.model.evals_result()
            self.genstats(self.model.evals_result(), i, time_now)
#            break

    def grid_search(self):
        self.model = xgboost.XGBClassifier(**self.params)
        xg_grid = GridSearchCV(
            self.model,
            {
                'learning_rate': [0.01, 0.5, 0.1, 0.2, 0.3]
            },
            cv=3,
            verbose=10,
            n_jobs=1
        )
        xg_grid.fit(self.X, self.y)
        print xg_grid.best_params_
        
        best_param, score, _ = max(xg_grid.grid_scores_, key=lambda x:x[1])
        print 'score:',score
        for param_name in sorted(best_param.keys()):
            print("%s: %r" % (param_name, best_param[param_name]))

    def predict_validation(self):
        self.model = xgboost.XGBClassifier(**self.params).fit(self.X, self.y)

        np.savetxt('imp_weigts.txt', np.array(self.model.feature_importances_))
        return
        
        print 'Predicting validation.'
        y_pred = self.model.predict_proba(self.X_valid)
        print y_pred
        print 'completed.. Predicting validation.'

        print 'Writing out validation.'
        wfile = open('temp.csv', 'w')
        wfile.write('qid,uid,label\n')
        for i,entry in enumerate(y_pred):
            wfile.write(str(self.validate_ids_dataframe['q_id'][i]) +',' + str(self.validate_ids_dataframe['u_id'][i]) +',{0:.20f}\n'.format(entry[1]))
        wfile.close()
        print 'Completed.. writing out validation.'

        print 'Predicting test.'
	y_pred = self.model.predict_proba(self.X_test)
        print 'Completed.. Predicting test.'

        print 'Writing out test.'
        wfile = open('final.csv', 'w')
        wfile.write('qid,uid,label\n')
        for i,entry in enumerate(y_pred):
            wfile.write(str(self.test_ids_df['q_id'][i]) +',' + str(self.test_ids_df['u_id'][i]) +',{0:.20f}\n'.format(entry[1]))
        wfile.close()
        print 'Completed.. Writing out test.'


if __name__ == '__main__':
    xg = xgboost_wrapper()
    xg.predict_validation()    
    xg.cross_validate()
#    xg.grid_search()



