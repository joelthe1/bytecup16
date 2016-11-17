import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.lda import LDA
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from scipy.stats import zscore

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
        
    def _count_vector(self, vocabulary, documents, ngram=(1,1)):
        return CountVectorizer(vocabulary=vocabulary, token_pattern=u'(?u)\\b\\w+\\b',
                               ngram_range=ngram).fit_transform(documents).toarray()
    
    def _tfidf(self, vocabulary, documents, ngram=(1, 1)):
        count_vectors = CountVectorizer(vocabulary=vocabulary, token_pattern=u'(?u)\\b\\w+\\b',
                                        ngram_range=ngram).fit_transform(documents)
        tf_vectors = TfidfTransformer(use_idf=True, norm='l2').fit_transform(count_vectors)
        return tf_vectors.todense()

    def _tag_vectors(self):
        '''
        Return tag features for users based on the training data.
        '''
        user_question_train_map_one = {}
        user_question_train_map_zero = {}
        tag_index_map = {tag:i for i, tag in enumerate(self.topic_vocabulary)}
        question_tag_map = {t['q_id']:str(t['q_tag']) for i,t in self.questions.iterrows()}

        for i, t in self.train.iterrows():
            if int(t['answered']) == 1:
                user_question_train_map_one.setdefault(t['u_id'], []).append(t['q_id'])
            else:
                user_question_train_map_zero.setdefault(t['u_id'], []).append(t['q_id'])

        user_tag_features = []
        for i, u in self.users.iterrows():
            questions_answered = user_question_train_map_one.get(u['u_id'], [])
            questions_not_answered = user_question_train_map_zero.get(u['u_id'], [])
            tag_vector_template = np.zeros(len(self.topic_vocabulary))
            for q in questions_answered:
                tag_vector_template[tag_index_map[question_tag_map[q]]] += 1
            for q in questions_not_answered:
                tag_vector_template[tag_index_map[question_tag_map[q]]] += -1
            user_tag_features.append(tag_vector_template)
 
        return normalize(axis=0, X=np.array(user_tag_features))

    def _user_question_median_score(self):
        '''
        Return the median of upvotes,ans,no_quality_ans, the questions answered by a user.
        '''
        user_question_train_map_one = {}
        user_question_train_map_zero = {}
        question_tag_map = {t['q_id']:[t['q_no_upvotes'],t['q_no_answers'],t['q_no_quality_answers']] for i,t in self.questions.iterrows()}
        for i, t in self.train.iterrows():
            if int(t['answered']) == 1:
                user_question_train_map_one.setdefault(t['u_id'], []).append(question_tag_map[t['q_id']])
            else:
                user_question_train_map_zero.setdefault(t['u_id'], []).append(question_tag_map[t['q_id']])
        user_question_median_features = []
        for i, u in self.users.iterrows():
            questions_answered = user_question_train_map_one.get(u['u_id'], [[0,0,0]])
            questions_answered = np.median(np.array(questions_answered), axis=0)
            questions_not_answered = user_question_train_map_zero.get(u['u_id'], [[0,0,0]])
            questions_not_answered = np.median(np.array(questions_not_answered), axis=0)
            user_features_template = np.hstack([questions_answered,questions_not_answered])
            user_question_median_features.append(user_features_template)

        return normalize(axis=0, X=np.array(user_question_median_features))
            
    def user_tag_vectors_dict(self):
        '''
        return user tag vector map
        '''
        user_tag_vector_map = {}
        tag_vectors = self._tag_vectors()
        assert tag_vectors.shape[0] == len(self.users)
        for i, u in self.users.iterrow():
            user_tag_vector_map[u['u_id']] = tag_vectors[i, :]
        return user_tag_vector_map
                

    def weighted_word_vectors(self):
        '''
        Return tf-idf count vectors back.
        '''
        question_word_tfidf_vectors = self._tfidf(self.word_vocabulary, self.questions['q_word_seq'].tolist(), (1, 2))
        numeric_question_vectors = normalize(axis=0, X=self.questions.as_matrix(['q_no_upvotes', 'q_no_answers',
                                                                                        'q_no_quality_answers']))
        question_feature_matrix = np.hstack([question_word_tfidf_vectors, numeric_question_vectors])
        user_word_tfidf_vector = self._tfidf(self.word_vocabulary, self.users['e_desc_word_seq'].tolist(), (1, 2))
        user_tag_vectors = self._tag_vectors()
        user_feature_matrix = np.hstack([user_word_tfidf_vector, user_tag_vectors])
        return question_feature_matrix, user_feature_matrix

    def nmf(self, components=100):
        '''
        Returns question_feature_dict and user_feature_dict
        '''

        question_word_vectors = self._tfidf(self.word_vocabulary, self.questions['q_word_seq'].tolist(), (1, 2))
        print 'running question word NMF...'        
        question_nmf_matrix = NMF(n_components=components).fit_transform(question_word_vectors)
        numeric_question_vectors = normalize(axis=0, X = self.questions.as_matrix(['q_no_upvotes', 'q_no_answers',
                                                                                        'q_no_quality_answers']))
        question_feature_matrix = np.hstack([question_nmf_matrix, numeric_question_vectors])
        user_word_tfidf_vector = self._tfidf(self.word_vocabulary, self.users['e_desc_word_seq'].tolist(), (1, 2))
        user_tag_vectors = self._tag_vectors()
        print 'running user word NMF...'
        user_nmf_matrix = NMF(n_components=components).fit_transform(user_word_tfidf_vector)
        user_feature_matrix = np.hstack([user_tag_vectors, user_nmf_matrix])
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
        numeric_question_vectors = normalize(axis=0, X=self.questions.as_matrix(['q_no_upvotes', 'q_no_answers',
                                                                                        'q_no_quality_answers']))
        print 'running question PCA...'
        question_feature_matrix = PCA(n_components=components[0]).fit_transform(np.hstack([question_word_tfidf_vectors,
                                                                                        numeric_question_vectors]))
        user_word_tfidf_vectors = self._tfidf(self.word_vocabulary, self.users['e_desc_word_seq'].tolist(), (1, 2))
        user_tag_vectors = self._tag_vectors()
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
    data._user_question_median_score()
    data._tags_vectors()
    data.pca()
    data.nmf()
