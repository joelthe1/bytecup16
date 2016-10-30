import numpy as np
import pandas as pd
import sys
import copy
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

###
#prepare data
###

class train_model:
    def __init__(self):
        self.question_dataframe = None
        self.user_dataframe = None
        self.train_info_dataframe = None

        self.word_vocabulary = None
        self.char_vocabulary = None
        self.topic_vocabulary = None

        self.cv_word = None
        self.cv_char = None
        self.cv_topic = None

        self.X = None #stored as a CSR Matrix
        self.y = None #stored as an array

    def load_data(self):
        ###
        # 1. Load Data
        ###
        print "\nLoading Data..."
        question_path = "bytecup2016data/question_info.txt"
        user_path = "bytecup2016data/user_info.txt"
        invited_info_path = "bytecup2016data/invited_info_train.txt"

        q_column_names = ['q_id', 'q_tag', 'q_word_seq', 'q_char_seq', 'q_no_upvotes', 'q_no_answers', 'q_no_quality_answers']
        u_column_names = ['u_id','e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']
        train_info_column_names = ['q_id','u_id','answered']

        self.question_dataframe = pd.read_csv(question_path, names=q_column_names, sep = '\t')
        self.user_dataframe = pd.read_csv(user_path, names = u_column_names, sep = '\t')
        self.train_info_dataframe = pd.read_csv(invited_info_path, names = train_info_column_names, sep = '\t')
        print "Loading Data (complete)..."

    def prepare_vocabulary(self):
        ###
        # 2. Prepare Vocabulary
        ###

        print "\nPreparing Vocabulary..."
        all_word_desc_list = self.question_dataframe['q_word_seq'].tolist() + self.user_dataframe['e_desc_word_seq'].tolist()
        all_char_desc_list = self.question_dataframe['q_char_seq'].tolist() + self.user_dataframe['e_desc_char_seq'].tolist()
        all_topics_list = self.question_dataframe['q_tag'].tolist() + self.user_dataframe['e_expert_tags'].tolist()

        self.word_vocabulary = set([word for sent in all_word_desc_list for word in str(sent).split('/')])
        self.char_vocabulary = set([char for sent in all_char_desc_list for char in str(sent).split('/')])
        self.topic_vocabulary = set([char for sent in all_topics_list for char in str(sent).split('/')])
        print "Size of the word vocabulary :", len(self.word_vocabulary)
        print "Size of the char vocabulary :", len(self.char_vocabulary)
        print "Number of topics : ", len(self.topic_vocabulary)
        print "Preparing Vocabulary (complete)..."

    def vectorize_data(self):
        ###
        # 3. vectorize data
        ###
        print "\nVectorizing data..."
        self.cv_word = CountVectorizer(vocabulary=self.word_vocabulary, token_pattern=u'(?u)\\b\\w+\\b')
        self.cv_char= CountVectorizer(vocabulary=self.char_vocabulary, token_pattern=u'(?u)\\b\\w+\\b')
        self.cv_topic = CountVectorizer(vocabulary=self.topic_vocabulary, token_pattern=u'(?u)\\b\\w+\\b')
        print "Vectorizing data (complete)..."


    def combine_data(self):
        ###
        # 4. combine data to 1 sparse matrix
        ###
        print "\ncombining data..."
        q_dict = dict()
        u_dict = dict()
        print "\tcompiling questions..."
        for idx, entry in self.question_dataframe.iterrows():
            #f1 = cv_word.transform([re.sub("/"," ", entry['q_word_seq'])])
            f2 = self.cv_char.transform([re.sub("/"," ", entry['q_char_seq'])])
            f3 =  [entry['q_tag']]
            f4 = [entry['q_no_upvotes']]
            f5 = [entry['q_no_answers']]
            f6 = [entry['q_no_quality_answers']]
            q_dict[entry['q_id']] = csr_matrix(hstack([f2, f3, f4, f5, f6]))
            #q_dict[entry['q_id']] = csr_matrix(hstack([f1, f2, f3, f4, f5, f6]))

        print "\tcompiling users..."
        for idx, entry in self.user_dataframe.iterrows():
            u_column_names = ['u_id','e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']
            #f1 = cv_word.transform([re.sub("/", " ", entry['e_desc_word_seq'])])
            f2 = self.cv_char.transform([re.sub("/", " ", entry['e_desc_char_seq'])])
            f3 = self.cv_topic.transform([re.sub("/", " ", entry['e_expert_tags'])])
            u_dict[entry['u_id']] = csr_matrix(hstack([f2, f3]))
            #u_dict[entry['u_id']] = csr_matrix(hstack([f1, f2, f3]))

        self.X = list()
        tempX = list()
        self.y = list()
        print "\tcompiling training info..."
        for idx, entry in self.train_info_dataframe.iterrows():
            tempX.append(csr_matrix(hstack([q_dict[entry['q_id']], u_dict[entry['u_id']]])))
            self.y.append(entry['answered'])

        self.X = csr_matrix(vstack(tempX))
        print "combining data (complete)..."

    def save_sparse_csr(self,filename,array):
        np.savez(filename,data = array.data ,indices=array.indices, indptr =array.indptr, shape=array.shape )

    def load_sparse_csr(self,filename):  
        loader = np.load(filename)
        return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),shape = loader['shape'])

    def save_numpy_mat(self, filename, X):
        np_mat_file = open(filename, 'w')
        np.save(np_mat_file, X.toarray(), allow_pickle=True)
        np_mat_file.close()

    def save_data(self):
        ###
        # 5. store the matrices
        ###
        print "\nsaving  data to file..."
        #self.save_numpy_mat("np_mat.dat", self.X)
        self.save_sparse_csr("csr_mat.dat", self.X)
        print "saving  data to file(complete)..."

    def load_data(self):
        print "\nLoading data from file..."
        self.X = self.load_sparse_csr("csr_mat.dat.npz")
        print "Loading data from file(complete)..."

    def train_naive_bayes(self):
        ###
        # 6. train a Naive Bayes
        ###
        print "\ntraining Naive Bayes..."
        gnb = GaussianNB()
        gnb_model = gnb.fit(self.X.toarray(), self.y)
        print "training Naive Bayes (complete)..."
        return gnb_model

    def train_logistic_regression(self):
        ###
        # 6. train a logistic regression
        ###
        print "\ntraining logistic regression..."
        lr = LogisticRegression()
        lr_model = lr.fit(self.X.toarray(), self.y)
        print "training logistic regression (complete)..."
        return lr_model


    def save_model(self, model):
        ###
        # 7. save the trained model to file
        ###
        print "\saving naive bayes trained model to file..."
        joblib.dump(model, 'model.pkl') 
        print "\saving naive bayes trained model to file(complete)..."
        #clf = joblib.load('filename.pkl')  #to load saved model

    def load_model(self):
        model = joblib.load('model.pkl')  #to load saved mod
        return model

    def predict(self,model,X):
        print "\npredicting and saving results..."
        res = model.predict_proba(X.toarray())
        #temp
        predictions = open('predictions.txt','w')
        for entry in res:
            predictions.write(str(entry[0]) + ', '+str(entry[1])+'\n')
        predictions.close()
        print "\npredicting and saving results(complete)..."
    #def get_important_words():
    #    pass    

c = train_model()
#c.load_data()
#c.prepare_vocabulary()
#c.vectorize_data()
#c.combine_data()
#c.save_data()

#model = train_naive_bayes()
#model = c.train_logistic_regression()
#c.save_model(model)
model = c.load_model()
c.load_data()
c.predict(model, c.X)
