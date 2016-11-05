import pandas as pd
import numpy as np
import xgboost
from sklearn import cross_validation
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
        self.X = self.load_sparse_csr("../../data/csr_mat_train.dat.npz")
        self.X_test = self.load_sparse_csr("../../data/csr_mat_test.dat.npz").toarray()
        self.y = pickle.load(open("../../data/csr_mat_train_y.pkl",'r'))
        self.test_ids_dataframe = pd.read_pickle("../../data/validate_nolabel.pkl")

        test_size = 0.25
        seed = 7
        #self.X, self.X_dev, self.y, self.y_dev = cross_validation.train_test_split(self.X, self.y, test_size=test_size, random_state=seed)
        print "Loading data from file(complete)..."

    def train_xgboost(self):
        # fit model no training data
        self.model = xgboost.XGBClassifier(max_depth=10, n_estimators=300, learning_rate=0.03, silent=True, objective='binary:logistic', gamma=2, min_child_weight=4, max_delta_step=5, subsample=0.8, reg_lambda=3, reg_alpha=1, scale_pos_weight=8.17).fit(self.X, self.y)
        print(self.model)

    def predict(self):
        # make predictions for test data
        y_pred = self.model.predict_proba(self.X_test)
        wfile = open('temp.csv', 'w')
        wfile.write('qid,uid,label\n')
        print y_pred
        for i,entry in enumerate(y_pred):
            wfile.write(str(self.test_ids_dataframe['q_id'][i]) +',' + str(self.test_ids_dataframe['u_id'][i]) +','+str(entry[1])+'\n')
#        print y_pred.shape

        
        #predictions = [round(value) for value in y_pred]
        
        # evaluate predictions
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



