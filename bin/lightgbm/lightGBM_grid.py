import sys
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier
sys.path.append('../')
from feature_engineering.load_data import loadData

execs = "~/LightGBM/lightgbm"
data = loadData('../../data')

X, Y, Xval, Xtest = data.dataset()

gbm = GBMClassifier(exec_path=execs)

param_grid = {'application':['binary'],
              'learning_rate': [0.05],
              'bagging_fraction': [1, 0.8],
              'num_iterations': [550],
              'num_leaves' : [50, 127],
              'lambda_l1':[0, .3],
              'lambda_l2':[0, .3],
              'max_depth':[8, 10, 12],
              'max_bin':[255],
              'feature_fraction':[1],
              'tree_learner':['serial', 'feature', 'data'],
              'boosting_type':['dart'],
              'bagging_fraction':[.8],
              'bagging_freq':[10],
              'early_stopping_round':[10],
              'drop_rate':[0.01],
              'num_threads':[24]
              }
              
scorer = metrics.make_scorer(metrics.roc_auc_score, greater_is_better=True)
clf = model_selection.GridSearchCV(gbm, param_grid, scoring=scorer, cv=5)
clf.fit(X,Y)
print("Best params: ", clf.best_params_)



# write test file
y_pred = clf.predict(Xtest)
with open('test_output.txt','w') as f:
    f.write('qid,uid,label\n')
    for i, test in data.test.iterrows():
        f.write('{},{},{}\n'.format(test['q_id'],test['u_id'],y_pred[i]))
# write valid file        
y_pred = clf.predict(Xval)
with open('valid_output.txt','w') as f:
    f.write('qid,uid,label\n')
    for i, test in data.validation.iterrows():
        f.write('{},{},{}\n'.format(test['q_id'],test['u_id'],y_pred[i]))






