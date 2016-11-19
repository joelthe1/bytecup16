import sys
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier
sys.path.append('../')
from feature_engineering.load_data import loadData

execs = "~/LightGBM/lightgbm"
data = loadData('../../data')

X, Y = data.training_features()

gbm = GBMClassifier(exec_path=execs,
                    metric='binary_error')

param_grid = {'learning_rate': [0.1, 0.04],
              'bagging_fraction': [0.5, 0.9],
              'num_iterations':[100, 150, 180, 250],
              'num_leaves' : [100, 127, 150],
              'lambda_l1':[0, .1],
              'lambda_l2':[0, .3, 3],
              'max_depth':[-1, 10, 12, 14],
              'max_bin':[255, 350, 450],
              'feature_fraction':[1, .6, .3, .1],
              'tree_learner':['serial', 'feature', 'data'],
              'boosting_type':['gbdt', 'dart'],
              'bagging_fraction':[1],
              'bagging_freq':[10],
              'early_stopping_round':[10, 15],
              'drop_rate':[0.01, 0.1, 0.03],
              'num_threads':[24]
              }
              
scorer = metrics.make_scorer(metrics.roc_auc_score, greater_is_better=True)
clf = model_selection.GridSearchCV(gbm, param_grid, scoring=scorer, cv=2)

clf.fit(X, Y)

print("Best score: ", clf.best_score_)
print("Best params: ", clf.best_params_)
