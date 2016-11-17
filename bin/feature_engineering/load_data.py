import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.lda import LDA
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF


class loadData:
    '''
        Load bytecup data from a path.
    '''

    def __init__(self, load_path):
        path = load_path
        questioncolumns = ['q_id', 'q_tag', 'q_word_seq', 'q_char_seq', 'q_no_upvotes', 'q_no_answers',
                           'q_no_quality_answers']
        usercolumns = ['u_id', 'e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']
        traincolumns = ['q_id', 'u_id', 'answered']

        self.questions = pd.read_csv(path + "/question_info.txt", names=questioncolumns, sep='\t')
        self.users = pd.read_csv(path + "/user_info.txt", names=usercolumns, sep='\t')
        self.train = pd.read_csv(path + "/invited_info_train.txt", names=traincolumns, sep='\t')
        self._vocabulary()

    def __repr__(self):
        return 'No of questions:{}\n No of users:{}\nNo of training examples:{}'.format(len(self.questions),
                                                                                        len(self.users), self.train)

    def _vocabulary(self):
        all_word_desc_list = self.questions['q_word_seq'].tolist() + self.users['e_desc_word_seq'].tolist()
        all_char_desc_list = self.questions['q_char_seq'].tolist() + self.users['e_desc_char_seq'].tolist()
        all_topics_list = self.questions['q_tag'].tolist() + self.users['e_expert_tags'].tolist()
        self.word_vocabulary = set([word for sent in all_word_desc_list for word in str(sent).split('/')])
        self.char_vocabulary = set([char for sent in all_char_desc_list for char in str(sent).split('/')])
        self.topic_vocabulary = set([char for sent in all_topics_list for char in str(sent).split('/')])

    def _tfidf(self, vocabulary, documents, ngram=(1, 1)):
        count_vectors = CountVectorizer(vocabulary=vocabulary, token_pattern=u'(?u)\\b\\w+\\b',
                                        ngram_range=ngram).fit_transform(documents)
        tf_vectors = TfidfTransformer(use_idf=True, norm='l2').fit_transform(count_vectors)
        return tf_vectors.todense()

    def nmf(self, components=100):
        '''
        Returns question_feature_dict and user_feature_dict
        '''

        question_word_tfidf_vectors = self._tfidf(self.word_vocabulary, self.questions['q_word_seq'].tolist(), (1, 2))
        question_tag_vectors = self._tfidf(self.topic_vocabulary, self.questions['q_tag'].astype('str').tolist())
        print 'running question NMF...'        
        lda_tag_tfidf_vectors = NMF(n_components=components).fit_transform(
            np.hstack([question_tag_vectors, question_word_tfidf_vectors]))
        numeric_question_vectors = Normalizer().fit_transform(self.questions.as_matrix(['q_no_upvotes', 'q_no_answers',
                                                                                        'q_no_quality_answers']))
        question_feature_matrix = np.hstack([lda_tag_tfidf_vectors, numeric_question_vectors])
        user_word_tfidf_vector = self._tfidf(self.word_vocabulary, self.users['e_desc_word_seq'].tolist(), (1, 2))
        user_tag_vectors = self._tfidf(self.topic_vocabulary, self.users['e_expert_tags'].tolist())
        print 'running user NMF...'
        user_feature_matrix = NMF(n_components=components).fit_transform(
            np.hstack([user_word_tfidf_vector, user_tag_vectors]))
        print 'question_shape:{}, user_shape:{}'.format(question_feature_matrix.shape, user_feature_matrix.shape)
        return question_feature_matrix, user_feature_matrix

    def nmf_array(self, components=100):
        '''
        return array features
        '''
        return self.nmf(components)
    
    def nmf_dict(self, components=100):
        '''
        return dict of nmf features
        '''
        question_feature_matrix, user_feature_matrix = self.nmf(components)
        user_feature_map = {}
        question_feature_map = {}
        for i, u in self.users.iterrows():
            user_feature_map[u['u_id']] = user_feature_matrix[i, :]
        for i, q in self.questions.iterrows():
            question_feature_map[q['q_id']] = question_feature_matrix[i, :]

        return question_feature_map, user_feature_map

    def pca(self, components=(4400, 4300)):
        '''
        Performs linear PCA and returns question_feature_dict and user_feature_dict
        '''
        question_word_tfidf_vectors = self._tfidf(self.word_vocabulary, self.questions['q_word_seq'].tolist(), (1, 2))
        question_tag_vectors = self._tfidf(self.topic_vocabulary, self.questions['q_tag'].astype('str').tolist())
        numeric_question_vectors = Normalizer().fit_transform(self.questions.as_matrix(['q_no_upvotes', 'q_no_answers',
                                                                                        'q_no_quality_answers']))
        print 'running question PCA...'
        question_feature_matrix = PCA(n_components=components[0]).fit_transform(np.hstack([question_word_tfidf_vectors,
                                                                                        question_tag_vectors,
                                                                                        numeric_question_vectors]))
        user_word_tfidf_vectors = self._tfidf(self.word_vocabulary, self.users['e_desc_word_seq'].tolist(), (1, 2))
        user_tag_vectors = self._tfidf(self.topic_vocabulary, self.users['e_expert_tags'].tolist())
        print 'running user PCA...'
        user_feature_matrix = PCA(n_components=components[1], svd_solver='randomized').fit_transform(
            np.hstack([user_word_tfidf_vectors, user_tag_vectors]))
        print 'question_shape:{}, user_shape:{}'.format(question_feature_matrix.shape, user_feature_matrix.shape)
        return question_feature_matrix, user_feature_matrix

    def pca_array(self, components=(4400, 4300)):
        return self.pca(components)
        
    def pca_dict(self, components=(4400, 4300)):
        '''
        return dict of question and user features.
        '''
        question_feature_matrix, user_feature_matrix = self.pca(components)
        user_feature_map = {}
        question_feature_map = {}
        for i, u in self.users.iterrows():
            user_feature_map[u['u_id']] = user_feature_matrix[i, :]
        for i, q in self.questions.iterrows():
            question_feature_map[q['q_id']] = question_feature_matrix[i, :]

        return question_feature_map, user_feature_map

    def raw_training_data(self):
        '''
        return training data in numpy.array format.
        '''
        Xtrain = []
        ytrain = []
        for i, t in self.train.iterrows():
            Xtrain.append([t['q_id'], t['u_id']])
            ytrain.append(int(t['answered']))
        return np.array(Xtrain), np.array(ytrain)

    def training_features(self, method='pca'):
        '''
        return question_features, user_features, labesl in np.array format 
        '''
        question_features = []
        user_features = []
        labels = []

        if method == 'pca':
            q_map, u_map = self.pca_dict()
        elif method == 'nmf':
            q_map, u_map = self.nmf_dict()
        else:
            print 'Error: method not defined'
            return
        
        for i, t in self.train.iterrows():
            labels.append(int(t['answered']))
            question_features.append(q_map[t['q_id']])
            user_features.append(u_map[t['u_id']])
        return np.array(question_features), np.array(user_features), np.array(labels)
    
if __name__ == '__main__':
    data = loadData('../../data')
    data.pca()
    data.nmf()
