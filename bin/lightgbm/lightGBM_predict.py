import sys
import numpy as np
from sklearn import datasets, metrics, model_selection
from pylightgbm.models import GBMClassifier
sys.path.append('../')
from feature_engineering.load_data import loadData

execs = "~/LightGBM/lightgbm"
data = loadData('../../data')

X, Y, Xval, Xtest = data.dataset()

gbm = GBMClassifier(exec_path=execs,
                    metric='binary_error',
                    learning_rate=.1,
                    num_iterations=200,
                    num_leaves=100,
                    lambda_l1=0,
                    lambda_l2=.3,
                    max_depth=10,
                    max_bin=255,
                    feature_fraction=.5,
                    tree_learner='serial',
                    boosting_type='dart',
                    bagging_fraction=.8,
                    bagging_freq=10,
                    early_stopping_round=10,
                    drop_rate=.1,
                    num_threads=24)

gbm.fit(X,Y)
y_pred = gbm.predict(Xtest)
with open('lgmb_output.txt','w') as f:
    f.write('qid,uid,label\n')
    for i, test in data.test.iterrows():
        f.write('{},{},{}\n'.format(test['q_id'],test['u_id'],y_pred[i]))


                    
                    
