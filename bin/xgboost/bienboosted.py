import numpy as np
import xgboost
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
import pickle

class xgboost_wrapper:
    def __init__(self):
        self.X = None #stored as a CSR Matrix
        self.y = None #stored as an array
        self.X_dev = None
        self.y_dev = None
        self.X_test = None

        self.model = None;

    def load_sparse_csr(self,filename):  
        loader = np.load(filename)
        return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),shape = loader['shape'])

    # load data
    def load_data(self):
        print "\nLoading data from file..."
        self.X = self.load_sparse_csr("../../data/csr_mat_train.dat.npz")
        self.X_test = self.load_sparse_csr("../../data/csr_mat_test.dat.npz")
        self.y = pickle.load(open("../../data/csr_mat_train_y.pkl",'r'))

        test_size = 0.33
        seed = 7
        self.X, self.X_dev, self.y, self.y_dev = cross_validation.train_test_split(self.X, self.y, test_size=test_size, random_state=seed)
        print "Loading data from file(complete)..."

    def train_xgboost(self):
        # fit model no training data
        self.model = xgboost.XGBClassifier()
        self.model.fit(self.X, self.y)
        print(self.model)

    def predict(self):
        # make predictions for test data
        y_pred = self.model.predict(self.X_dev)
        predictions = [round(value) for value in y_pred]
        
        # evaluate predictions
        accuracy = accuracy_score(self.y_dev, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

xg = xgboost_wrapper()
xg.load_data()
xg.train_xgboost()
xg.predict()


