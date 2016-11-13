import pandas as pd
import numpy as np
import xgboost
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import csr_matrix
import pickle

class xgboost_wrapper:
    def __init__(self):
        self.X = None #stored as a CSR Matrix
        self.y = None #stored as an array
        self.X_dev = None
        self.y_dev = None
        self.X_test = None

        self.test_ids_dataframe = None

        self.model = None;

    def load_sparse_csr(self,filename):  
        loader = np.load(filename)
        return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),shape = loader['shape'])

    # load data
    def load_data(self):
        print "\nLoading data from file..."
        self.X = self.load_sparse_csr("../../data/csr_mat_train_lsa.dat.npz").toarray()
        self.X_test = self.load_sparse_csr("../../data/csr_mat_test_lsa.dat.npz").toarray()
        self.y = pickle.load(open("../../data/csr_mat_train_y.pkl",'r'))
        self.test_ids_dataframe = pd.read_pickle("../../data/test_nolabel.pkl")

#        test_size = 0.25
#        seed = 7
#        self.X, self.X_dev, self.y, self.y_dev = cross_validation.train_test_split(self.X, self.y, test_size=test_size, random_state=seed)
        print "Loading data from file(complete)..."

    def fpreproc(self, dtrain, dtest, param):
        label = dtrain.get_label()
        ratio = float(np.sum(label == 0)) / np.sum(label==1)
        param['scale_pos_weight'] = ratio
        return (dtrain, dtest, param)

    def train_xgboost(self):
        # fit model no training data
        self.model = xgboost.XGBClassifier(max_depth=10, n_estimators=100, learning_rate=0.08, silent=True, objective='binary:logistic', gamma=0.2, min_child_weight=1, max_delta_step=6, subsample=0.8, reg_lambda=3, reg_alpha=1, scale_pos_weight=1).fit(self.X, self.y, eval_metric='ndcg@10')

#        dtrain = xgboost.DMatrix(self.X, label=self.y)
#        self.X = None
#        self.y = None
#        param = {'max_depth':10, 'n_estimators':1, 'learning_rate':0.08, 'silent':True, 'objective':'binary:logistic', 'gamma':0.2, 'min_child_weight':0, 'max_delta_step':6, 'subsample':0.8, 'reg_lambda':3, 'reg_alpha':1, 'scale_pos_weight':1}
#        res = xgboost.cv(param, dtrain, num_boost_round=10, nfold=5, stratified=True, metrics={'ndcg@10'}, seed = 0, callbacks=[xgboost.callback.print_evaluation(show_stdv=True)])
#        #fpreproc=self.fpreproc
#        print(res)

#        clf = GridSearchCV(
#            self.model,
#            {
#                'max_depth': [3, 6, 10],
#                'n_estimators': [10,50,100],
#                'min_child_weight': [1,3,6]
#            },
#            cv=10,
#            verbose=10
#        )
#        clf.fit(self.X, self.y)
#        best_param, score, _ = max(clf.grid_scores_, key=lambda x:x[1])
#        print 'score:',score
#        for param_name in sorted(best_param.keys()):
#            print("%s: %r" % (param_name, best_param[param_name]))

    def predict(self):
        # make predictions for test data
        y_pred = self.model.predict_proba(self.X_test)
        wfile = open('temp.csv', 'w')
        wfile.write('qid,uid,label\n')
        for i,entry in enumerate(y_pred):
            wfile.write(str(self.test_ids_dataframe['qid'][i]) +',' + str(self.test_ids_dataframe['uid'][i]) +','+str(entry[1])+'\n')
        print y_pred.shape

        
#        predictions = [round(value[1]) for value in y_pred]
#       
#       # evaluate predictions
#        accuracy = accuracy_score(self.y_dev, predictions)
#        f1 = f1_score(self.y_dev, predictions, labels=None, pos_label=1)
#        corr = 0
#        for i in range(len(predictions)):
#            if self.y_dev[i] == 1 and self.y_dev[i] == predictions[i]:
#                corr += 1
#        print("Accuracy: %.2f%%, f1: %.2f, correct: %d out of %d" % ((accuracy * 100.0), f1, corr, len(predictions)))

xg = xgboost_wrapper()
xg.load_data()
xg.train_xgboost()
xg.predict()
