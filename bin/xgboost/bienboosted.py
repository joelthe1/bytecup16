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

class xgboost_wrapper:
    def __init__(self):
        data = loadData('../../data')
        q_mat, u_mat, self.y = data.training_features()
        self.X = np.hstack([q_mat, u_mat])
        self.X_dev = None
        self.y_dev = None
        self.X_test = None
        self.validte_ids_dataframe = None

        self.model = None;
        
        self.params = {'max_depth':10,
                       'n_estimators':300,
                       'learning_rate':0.02,
                       'silent':True,
                       'objective':'binary:logistic',
                       'gamma':0.2,
                       'min_child_weight':1,
                       'max_delta_step':6,
                       'subsample':0.8,
                       'reg_lambda':3,
                       'reg_alpha':1,
                       'scale_pos_weight':1}

    def cross_validate(self):
        dtrain = xgboost.DMatrix(self.X, label=self.y)
        skf = StratifiedKFold(n_splits=5, random_state=2016)
        for train_index, test_index in skf.split(self.X, self.y):
            Xtrain, ytrain = self.X[train_index], self.y[train_index]
            Xtest, ytest = self.X[test_index], self.y[test_index]
            print Xtrain.shape, ytrain.shape

            self.model = xgboost.XGBClassifier(**self.params).fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)], eval_metric='auc', verbose=True)
            print self.model.evals_result()
            break

    def grid_search(self):
       xg_grid = GridSearchCV(
           self.model,
           {
               'max_depth': [3, 6, 10],
               'n_estimators': [10,50,100],
               'min_child_weight': [1,3,6]
           },
           cv=10,
           verbose=10
       )
       xg_grid.fit(self.X, self.y)
       best_param, score, _ = max(clf.grid_scores_, key=lambda x:x[1])
       print 'score:',score
       for param_name in sorted(best_param.keys()):
           print("%s: %r" % (param_name, best_param[param_name]))

    def predict_validation(self):
        self.model = xgboost.XGBClassifier(**self.params).fit(self.X, self.y)
        y_pred = self.model.predict_proba(self.X_dev)
        wfile = open('temp.csv', 'w')
        wfile.write('qid,uid,label\n')
        for i,entry in enumerate(y_pred):
            wfile.write(str(self.validate_ids_dataframe['q_id'][i]) +',' + str(self.validate_ids_dataframe['u_id'][i]) +','+str(entry[1])+'\n')

if __name__ == '__main__':
    xg = xgboost_wrapper()
    xg.cross_validate()

