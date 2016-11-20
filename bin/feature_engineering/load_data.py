import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from scipy.stats import zscore, kurtosis, skew
from scipy.spatial import distance
import scipy.sparse
from lightfm import LightFM

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
        self.common_map = None
        self._vocabulary()
        self.freq_dict = pickle.load(open('freq_dict.pkl', 'r'))

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
        count_vectors = CountVectorizer(vocabulary=vocabulary, token_pattern=u'(?u)\\b\\w+\\b', min_df=.10, max_df=.75,
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
            tag_vector_template = np.zeros(len(self.topic_vocabulary)+4)
            for q in questions_answered:
                tag_vector_template[tag_index_map[question_tag_map[q]]] += 1
            for q in questions_not_answered:
                tag_vector_template[tag_index_map[question_tag_map[q]]] += -1
            # add 3 features --- number of questions answered,
            # number of questions not answered,
            # total expert tags per user
            tag_vector_template[-4] = len(u['e_desc_word_seq'].split('/'))
            tag_vector_template[-3] = len(questions_answered)
            tag_vector_template[-2] = len(questions_not_answered)
            tag_vector_template[-1] = len(u['e_expert_tags'].split('/'))            
            user_tag_features.append(tag_vector_template)
        # find prior of answering for each user
        prior = []
        for u in self.users['u_id'].tolist():
            if u in user_question_train_map_one:
                prior.append(len(user_question_train_map_one[u])/float(len(user_question_train_map_one[u])
                                                                  +len(user_question_train_map_zero.get(u,[]))))
            else:
                prior.append(0.0)
        neg_prior = np.reshape(map(lambda x:1-x, prior), (len(prior),1))
        prior = np.reshape(prior, (len(prior),1))
        return np.hstack([normalize(axis=0, X=np.array(user_tag_features)),neg_prior,prior])

    def question_tag_features(self):
        tag_question_map = {}
        tag_total_answers = {}
        for i,q in self.questions.iterrows():
            tag_question_map.setdefault(q['q_tag'], []).append([q['q_no_upvotes'],
                                                                q['q_no_answers'],q['q_no_quality_answers']])
            tag_total_answers.setdefault(q['q_tag'], 0)
            tag_total_answers[q['q_tag']] += 1
        tag_total_array = []
        tag_feature_matrix = []
        for tag in self.questions['q_tag'].tolist():
            tag_features = np.array(tag_question_map[tag])
            tag_features_template = np.hstack([np.mean(tag_features, axis=0),
                                               np.median(tag_features, axis=0),
                                               np.ndarray.min(tag_features, axis=0),
                                               np.ndarray.max(tag_features, axis=0),
                                               kurtosis(tag_features, axis=0)])
            tag_feature_matrix.append(tag_features_template)
            tag_total_array.append(tag_total_answers[tag])

        #find the prior of answering each question
        question_count_map = Counter(self.train['q_id'].tolist())
        question_ans_count_map = {}
        prior = []
        for i,t in self.train.iterrows():
            question_ans_count_map[t['q_id']] = question_ans_count_map.get(t['q_id'],0)+int(t['answered'])
        for q in self.questions['q_id'].tolist():
            if q in question_ans_count_map:
                prior.append(question_ans_count_map[q]/float(question_count_map[q]))
            else:
                prior.append(0.0)
        prior = np.reshape(prior, (len(prior),1))
        
        return np.hstack([np.array(tag_feature_matrix), prior,
                          np.reshape(tag_total_array, (len(self.questions),1))])

    def user_question_score(self):
        '''
        Return the median of upvotes,ans,no_quality_ans, the questions answered by a user.
        '''
        user_question_train_map_one = {}
        user_question_answered ={}
        user_question_train_map_zero = {}
        
        question_tag_map = {t['q_id']:[t['q_no_upvotes'],t['q_no_answers'],t['q_no_quality_answers']] for i,t in self.questions.iterrows()}
        for i, t in self.train.iterrows():
            if int(t['answered']) == 1:
                user_question_train_map_one.setdefault(t['u_id'], []).append(question_tag_map[t['q_id']])
                user_question_answered.setdefault(t['u_id'],[]).append(t['q_id'])
            else:
                user_question_train_map_zero.setdefault(t['u_id'], []).append(question_tag_map[t['q_id']])
        user_question_median_features = []
        for i, u in self.users.iterrows():
            q_answered = np.array(user_question_train_map_one.get(u['u_id'], [[0,0,0]]))
            q_not = np.array(user_question_train_map_zero.get(u['u_id'], [[0,0,0]]))
            user_features_template = np.hstack([np.median(q_answered, axis=0),
                                                np.mean(q_answered, axis=0),
                                                skew(q_answered, axis=0),
                                                np.ndarray.min(q_answered, axis=0),
                                                np.ndarray.max(q_answered, axis=0),
                                                kurtosis(q_answered, axis=0),
                                                np.median(q_not, axis=0),
                                                np.mean(q_not, axis=0),
                                                np.ndarray.min(q_not, axis=0),
                                                np.ndarray.max(q_not, axis=0),
                                                skew(q_not, axis=0),
                                                kurtosis(q_not, axis=0)])
            user_question_median_features.append(user_features_template)
        #create a common map if not present
        count_out_domain = []
        self._do_share_tag(self.train['u_id'].tolist(), self.train['q_id'].tolist())
        for u in self.users['u_id'].tolist():
            count_temp = 0
            for q in user_question_answered.get(u,[]):
                if u in self.common_map and q in self.common_map[u] and self.common_map[u][q] != 1:
                    count_temp += (1.0/len(user_question_answered[u]))
            count_out_domain.append(count_temp)
        count_out_domain = np.reshape(count_out_domain, (len(self.users),1))
        return np.hstack([normalize(axis=0, X=np.array(user_question_median_features)),count_out_domain])
            
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

    def _do_share_tag(self, users, questions):
        assert len(users) == len(questions)
        if self.common_map is not None:
            print 'using pre-computed common-tag features...'
            do_share_info = []
            for i in range(len(users)):
                do_share_info.append(self.common_map[users[i]][questions[i]])
            print '----->finished.'
            return np.reshape(do_share_info, (len(users),1))
        print 'computing common-tag features...'
        self.common_map = {}

        user_tag_map = {}
        question_tag_map = {}
        for i,u in self.users.iterrows():
            user_tag_map[u['u_id']] = u['e_expert_tags'].split('/')
        for i,q in self.questions.iterrows():
            question_tag_map[q['q_id']] = str(q['q_tag'])

        reduced_combinations = np.vstack([self.train.as_matrix(['u_id', 'q_id']),
                                         self.validation.as_matrix(['u_id', 'q_id']),
                                         self.test.as_matrix(['u_id', 'q_id'])])
        print '\tshape of reduced_combinations:',reduced_combinations.shape
        for u,q in reduced_combinations:
            self.common_map.setdefault(u,{})[q] = int(question_tag_map[q] in user_tag_map[u])
        do_share_info = []
        for i in range(len(users)):
           do_share_info.append(self.common_map[users[i]][questions[i]])
        return np.reshape(do_share_info, (len(users),1))

    def get_id_rel(self, i, x):
        max_val = 0
        min_val = 0
        log_val = 0
        for v in x:
            if v not in self.freq_dict[i]:
                val = 0
            else:
                val = self.freq_dict[i][v]
            if val > max_val:
                max_val = val
            if val < min_val:
                min_val = val
            log_val += np.log(1 + val)
        return [max_val, min_val, log_val]

    def qtag_utag_correlation(self, users, questions):
        user_tag_map = {}
        question_tag_map = {}
        assert len(users)==len(questions)
        for i,u in self.users.iterrows():
            user_tag_map[u['u_id']]= u['e_expert_tags'].split('/')
        for i,q in self.questions.iterrows():
            question_tag_map[q['q_id']]=str(q['q_tag'])
        proba = []
        for i in range(len(users)):
            proba.append(self.get_id_rel(question_tag_map[questions[i]], user_tag_map[users[i]]))
        return np.array(proba)
        
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

    def user_cosine_similarity(self, components=1000):
        tfidf_vectors = self._tfidf(self.word_vocabulary, self.users['e_desc_word_seq'].tolist(), (1, 1))
        user_similarity = (tfidf_vectors*tfidf_vectors.T).A
        return PCA(n_components=components, svd_solver='randomized').fit_transform(np.absolute(user_similarity))

    def question_cosine_similarity(self):
        tfidf_vectors = self._tfidf(self.word_vocabulary, self.questions['q_word_seq'].tolist(), (1, 1))
        question_similarity = (tfidf_vectors*tfidf_vectors.T).A
        return np.absolute(question_similarity)

    def latent_factors(self):
        '''
        returns question, user latent factors using LightFM
        '''
        print '\tcalculating latent factors...'
        q_index_map = {q['q_id']:i  for i,q in self.questions.iterrows()}
        u_index_map = {u['u_id']:i  for i,u in self.users.iterrows()}
        X = self.train.as_matrix(['q_id','u_id'])
        y = np.array(self.train['answered'].tolist())
        row = []
        column = []
        data = []
        for i, XV in enumerate(X):
            row.append(u_index_map[XV[1]])
            column.append(q_index_map[XV[0]])
            data.append(int(y[i]))
        train_matrix = scipy.sparse.csr_matrix((data, (row, column)),
                                               shape=(len(self.users),len(self.questions)))
        model = LightFM(loss='warp-kos', max_sampled=12, no_components=10)
        model.fit(train_matrix, epochs=15, num_threads=24)
        return model.user_embeddings, model.item_embeddings
    
    def tag_tag_similarity(self):
        tag_word_list_map = {}
        for i,q in self.questions.iterrows():
            tag_word_list_map[q['q_tag']] = q['q_word_seq']+'/'+tag_word_list_map.setdefault(q['q_tag'], '999999999')
        tag_word_documents = []
        for i,q in self.questions.iterrows():
            tag_word_documents.append(tag_word_list_map[q['q_tag']])
        tfidf_vectors = self._tfidf(self.word_vocabulary, tag_word_documents)
        tag_tag_similarity = (tfidf_vectors*tfidf_vectors.T).A
        return tag_tag_similarity

    def user_question_similarity(self, users, questions):
        assert len(questions)==len(users)
        user_word_features = self._count_vector(self.word_vocabulary, self.users['e_desc_word_seq'])
        question_word_features = self._count_vector(self.word_vocabulary, self.questions['q_word_seq'])
        q_index_map = {q['q_id']:i  for i,q in self.questions.iterrows()}
        u_index_map = {u['u_id']:i  for i,u in self.users.iterrows()}
        values = []
        for i in range(len(users)):
            values.append(distance.correlation(user_word_features[u_index_map[users[i]]],
                                   question_word_features[q_index_map[questions[i]]]))                          
        return np.reshape(values, (len(users),1))
        
    def dataset(self):
        '''
        return Xtrain, y_train, Xval, Xtest
        '''

        # -------- USER FEATURES ---------------
        print 'calculating user features...'
        user_latent_features, question_latent_features = self.latent_factors()
        user_features = np.hstack([self.user_question_score(),
                                   self.user_tag_vectors()])
        user_char_desc_len = np.reshape([len(char_des.split('/'))
                                         for char_des in self.users['e_desc_char_seq'].tolist()],
                                        (len(self.users),1))
        user_features = np.hstack([user_features, user_latent_features, user_char_desc_len])
        
        # -------- QUESTION FEATURES --------------
        print 'calculating question features...'
        question_features = normalize(self.questions.as_matrix(['q_no_upvotes',
                                                                'q_no_answers',
                                                                'q_no_quality_answers']),axis=0)
        question_tag_vectors = np.array(self._count_vector(self.topic_vocabulary,
                                                           map(lambda x:str(x), self.questions['q_tag'].tolist())))
        question_length_vectors = np.reshape([ len(each.split('/'))
                                               for each in self.questions['q_word_seq'].tolist()]
                                    , (len(self.questions),1))
        question_quality_vectors = np.reshape([q/float(a) if a>0 else 0.0 for a,q in self.questions.as_matrix(['q_no_answers', 'q_no_quality_answers'])],
                                              (len(self.questions),1))
        question_char_desc_len = np.reshape([len(desc.split('/'))
                                             for desc in self.questions['q_char_seq']],
                                            (len(self.questions),1))
        question_features = np.hstack([question_features,question_tag_vectors,
                                       question_char_desc_len,
                                       question_length_vectors,
                                       self.question_tag_features(),
                                       question_quality_vectors])
        
        question_features = np.hstack([question_features, self.question_cosine_similarity()])
        question_features = np.hstack([question_features, question_latent_features])
        question_features = np.hstack([question_features, self.tag_tag_similarity()])

        user_feat_map = {u['u_id']:i for i,u in self.users.iterrows()}
        question_feat_map = {q['q_id']:i for i,q in self.questions.iterrows()}

        def _X_features(data):
            print '\tcompiling features started... shape:',data.shape
            c_user_features = []
            c_question_features = []
            for i, t in data.iterrows():
                c_question_features.append(question_features[question_feat_map[t['q_id']], :])
                c_user_features.append(user_features[user_feat_map[t['u_id']], :])
            #  -------- COMMON FEATURES ---------
            print '\t\tcalculating user_question_similarity...'
            user_question_similarity = self.user_question_similarity(data['u_id'].tolist(), data['q_id'].tolist())
            print '\t\tcalculating common_tag_info...'
            common_tag_info = self._do_share_tag(data['u_id'].tolist(), data['q_id'].tolist())
            print '\t\tcalculating qtag_utag correlations...'
            qtag_utag_corr = self.qtag_utag_correlation(data['u_id'].tolist(), data['q_id'].tolist())
            return np.hstack([np.array(c_question_features),
                              qtag_utag_corr,
                              np.array(c_user_features), common_tag_info, user_question_similarity])
        
        print '----Xtrain----'
        Xtrain = _X_features(self.train)
        print '----Xval------'
        Xval = _X_features(self.validation)
        print '----Xtest-----'
        Xtest = _X_features(self.test)
        ytrain = np.array(map(lambda x:int(x), self.train['answered'].tolist()))
        
        return Xtrain, ytrain, Xval, Xtest

    def dataset_with_preprocessing(self):
        Xtrain, ytrain, Xval, Xtest = self.dataset()
        print 'removing low quality features...'
        # columns_to_consider = []
        # for i in range(len(Xtrain[0])):
        #     if len(np.unique(Xtrain[:,i])) > 1:
        #         columns_to_consider.append(i)
        # features based on weights from XGBOOST
        columns_to_consider =[0, 1, 2, 3, 5, 16, 27, 38, 53, 54, 57, 58, 69, 80, 91, 102,
                              113, 124, 135, 146, 147, 148, 149, 150, 151, 152, 153, 155,
                              157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
                              169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
                              182, 183, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196,
                              197, 198, 201, 205, 208, 209, 214, 215, 221, 222, 235, 239, 241,
                              269, 271, 277, 296, 298, 304, 337, 8271, 8273, 8274, 8275, 8276,
                              8277, 8278, 8279, 8280, 8281, 8282, 8283, 8284, 8285, 8286, 8287,
                              8288, 8289, 8290, 8291, 8292, 8293, 8294, 8295, 8296, 8297, 8298,
                              8299, 8300, 8301, 8302, 8303, 8304, 8305, 8306, 8307, 8308, 8309, 8310, 8331, 8332,
                              8333, 8372, 8373, 8404, 8405, 8416, 8417, 8418, 8419, 8420, 8421, 8422, 8423, 8424, 8425, 8438,
                              8439, 8454, 8455, 8456, 8457, 8458, 8459, 8460, 8461, 8462, 8463, 8464, 8465, 8466, 8467, 8468, 8469, 8470, 8472]
        return Xtrain[:, columns_to_consider], ytrain, Xval[:, columns_to_consider], Xtest[:, columns_to_consider]
        
    
if __name__ == '__main__':
    data = loadData('../../data')
    xtrain, ytrain, xval, xtest = data.dataset_with_preprocessing()
    print 'Xtrain:{},ytrain:{}\nxval:{}\nxtest{}'.format(xtrain.shape, ytrain.shape, xval.shape, xtest.shape)


                                      

                                      

