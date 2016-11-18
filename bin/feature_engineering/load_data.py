import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from scipy.stats import zscore, kurtosis

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
        validcolumns = ['q_id', 'u_id']
        testcolumns = ['q_id', 'u_id']

        self.questions = pd.read_csv(path + "/question_info.txt", names=questioncolumns, sep='\t')
        self.users = pd.read_csv(path + "/user_info.txt", names=usercolumns, sep='\t')
        self.train = pd.read_csv(path + "/invited_info_train.txt", names=traincolumns, sep='\t')
        self.validation = pd.read_csv(path + "/validate_nolabel.txt", names=validcolumns, sep=',', skiprows=1)
        self.test = pd.read_csv(path + "/test_nolabel.txt", names=testcolumns, sep=',', skiprows=1)
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

    def user_tag_vectors(self):
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

    def user_question_score(self):
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
            q_answered = np.array(user_question_train_map_one.get(u['u_id'], [[0,0,0]]))
            q_not = np.array(user_question_train_map_zero.get(u['u_id'], [[0,0,0]]))
            user_features_template = np.hstack([np.median(q_answered, axis=0),
                                                np.mean(q_answered, axis=0),
                                                np.ndarray.min(q_answered, axis=0),
                                                np.ndarray.max(q_answered, axis=0),
                                                kurtosis(q_answered, axis=0),
                                                np.median(q_not, axis=0),
                                                np.mean(q_not, axis=0),
                                                np.ndarray.min(q_not, axis=0),
                                                np.ndarray.max(q_not, axis=0),
                                                kurtosis(q_not, axis=0)])
            user_question_median_features.append(user_features_template)
        return normalize(axis=0, X=np.array(user_question_median_features))
            
    def weighted_word_vectors(self, pca=None):
        '''
        Return tf-idf count vectors back.
        '''
        question_word_feature_vectors = self._tfidf(self.word_vocabulary, self.questions['q_word_seq'].tolist(), (1, 2))
        user_word_feature_vectors = self._tfidf(self.word_vocabulary, self.users['e_desc_word_seq'].tolist(), (1, 2))
        if pca is None:
            return question_word_feature_vectors, user_word_feature_vectors
        else:
            assert type(pca)==tuple
            question_word_features = PCA(n_components=pca[0]).fit_transform(question_word_feature_vectors)
            user_word_features = PCA(n_components=pca[1]).fit_transform(user_word_feature_vectors)
            return question_word_features, user_word_features

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

    def user_cosine_similarity(self):
        tfidf_vectors = self._tfidf(self.word_vocabulary, self.users['e_desc_word_seq'].tolist(), (1, 1))
        user_similarity = (tfidf_vectors*tfidf_vectors.T).A
        return np.absolute(user_similarity)

    def question_cosine_similarity(self):
        tfidf_vectors = self._tfidf(self.word_vocabulary, self.questions['q_word_seq'].tolist(), (1, 1))
        question_similarity = (tfidf_vectors*tfidf_vectors.T).A
        return np.absolute(question_similarity)

    
    def training_features(self, method='', other=None):
        '''
        return question_features, user_features, labesl in np.array format 
        '''
        user_features = np.hstack([self.user_question_score(),
                                   self.user_tag_vectors()])
        question_features = normalize(self.questions.as_matrix(['q_no_upvotes',
                                                           'q_no_answers',
                                                                'q_no_quality_answers','q_tag']),axis=0)
        question_features = np.hstack([question_features, self.question_cosine_similarity()])
        user_feat_map = {u['u_id']:i for i,u in self.users.iterrows()}
        question_feat_map = {q['q_id']:i for i,q in self.questions.iterrows()}

        # return the features for validation or test files
        if other is not None:
            c_user_features = []
            c_question_features = []
            data = self.test if other is 'test' else self.validation
            for i, t in data.iterrows():
                c_question_features.append(question_features[question_feat_map[t['q_id']], :])
                c_user_features.append(user_features[user_feat_map[t['u_id']], :])
            return np.array(c_question_features), np.array(c_user_features)
                
        train_user_features = []
        train_question_features = []
        labels = []
        for i, t in self.train.iterrows():
            train_question_features.append(question_features[question_feat_map[t['q_id']], :])
            train_user_features.append(user_features[user_feat_map[t['u_id']], :])
            labels.append(int(t['answered']))        
        return np.array(train_question_features), np.array(train_user_features), np.array(labels)
    
if __name__ == '__main__':
    data = loadData('../../data')
    data.user_cosine_similarity()
    data.question_cosine_similarity()
    data.training_features()

                                      

                                      

