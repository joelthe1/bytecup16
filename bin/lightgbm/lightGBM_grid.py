import sys
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier
sys.path.append('../')
from feature_engineering.load_data import loadData



execs = "~/LightGBM/lightgbm"
data = loadData('../../data')

X, Y, Xval, Xtest = data.dataset()
print 'data loading complete....'
gbm = GBMClassifier(exec_path=execs)

param_grid = {'application':['binary'],
              'learning_rate': [0.08],
              'bagging_fraction': [1, 0.8],
              'num_iterations': [550],
              'num_leaves' : [127, 150],
              'lambda_l1':[.3],
              'max_depth':[8],
              'max_bin':[255],
              'feature_fraction':[1, .8],
              'tree_learner':['serial'],
              'boosting_type':['dart'],
              'bagging_fraction':[.8],
              'bagging_freq':[10],
              'early_stopping_round':[10],
              'drop_rate':[0.01],
              'num_threads':[24]
              }
              
scorer = metrics.make_scorer(metrics.roc_auc_score, greater_is_better=True)
print 'starting grid search....'
clf = model_selection.GridSearchCV(gbm, param_grid, scoring=scorer, cv=5)
clf.fit(X,Y)



print("Best params: ", clf.best_params_)
print("Best score: ", clf.best_score_)


print 'writing output files...'
# write test file
y_pred = clf.predict_proba(Xtest)
with open('test_output.txt','w') as f:
    f.write('qid,uid,label\n')
    for i, test in data.test.iterrows():
        f.write('{},{},{}\n'.format(test['q_id'],test['u_id'],y_pred[i]))
# write valid file        
y_pred = clf.predict_proba(Xval)
with open('valid_output.txt','w') as f:
    f.write('qid,uid,label\n')
    for i, test in data.validation.iterrows():
        f.write('{},{},{}\n'.format(test['q_id'],test['u_id'],y_pred[i]))






