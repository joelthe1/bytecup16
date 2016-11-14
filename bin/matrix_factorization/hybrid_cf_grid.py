import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import auc_score
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Normalizer

NUM_THREADS = 24
NUM_COMPONENTS = 320
NUM_EPOCHS = 20
ITEM_ALPHA = 1e-6

question_path = "../../data/question_info.txt"
user_path = "../../data/user_info.txt"
invited_info_path = "../../data/invited_info_train.txt"

q_column_names = ['q_id', 'q_tag', 'q_word_seq', 'q_char_seq', 'q_no_upvotes', 'q_no_answers', 'q_no_quality_answers']
u_column_names = ['u_id', 'e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']
train_info_column_names = ['q_id', 'u_id', 'answered']

question_dataframe = pd.read_csv(question_path, names=q_column_names, sep='\t')
user_dataframe = pd.read_csv(user_path, names=u_column_names, sep='\t')
train_info_dataframe = pd.read_csv(invited_info_path, names=train_info_column_names, sep='\t')

all_word_desc_list = question_dataframe['q_word_seq'].tolist() + user_dataframe['e_desc_word_seq'].tolist()
all_char_desc_list = question_dataframe['q_char_seq'].tolist() + user_dataframe['e_desc_char_seq'].tolist()
all_topics_list = question_dataframe['q_tag'].tolist() + user_dataframe['e_expert_tags'].tolist()

word_vocabulary = set([word for sent in all_word_desc_list for word in str(sent).split('/')])
char_vocabulary = set([char for sent in all_char_desc_list for char in str(sent).split('/')])
topic_vocabulary = set([char for sent in all_topics_list for char in str(sent).split('/')])

cv_word = CountVectorizer(vocabulary=word_vocabulary, token_pattern=u'(?u)\\b\\w+\\b', ngram_range=(1, 2))
cv_char = CountVectorizer(vocabulary=char_vocabulary, token_pattern=u'(?u)\\b\\w+\\b', ngram_range=(2, 5))
cv_tag = CountVectorizer(vocabulary=topic_vocabulary, token_pattern=u'(?u)\\b\\w+\\b')

word_counts = cv_word.fit_transform(question_dataframe['q_word_seq'].tolist())
tf_word = TfidfTransformer(use_idf=True, norm='l2').fit(word_counts)
word_tfs = tf_word.transform(word_counts)
tsvd = TruncatedSVD(n_components=150, n_iter=7, random_state=42)
word_tfs = tsvd.fit_transform(word_tfs)

char_counts = cv_char.fit_transform(question_dataframe['q_char_seq'].tolist())
tf_char = TfidfTransformer(use_idf=True, norm='l2').fit(char_counts)
char_tfs = tf_char.transform(char_counts)
tsvd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
char_tfs = tsvd.fit_transform(char_tfs)

tag_counts = cv_tag.fit_transform([str(i) for i in question_dataframe['q_tag'].tolist()])
tf_tags = TfidfTransformer(use_idf=True, norm='l2').fit(tag_counts)
tag_tfs = tf_tags.transform(tag_counts)
tsvd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
tag_tfs = tsvd.fit_transform(tag_tfs)

question_feature_matrix = Normalizer(copy=True).fit_transform(np.hstack([word_tfs, char_tfs, tag_tfs]))

question_feature_map = {}
question_features = []
for i, q in question_dataframe.iterrows():
    # question_feature_map[q['q_id']] = np.hstack([question_feature_matrix[i,:], q['q_no_upvotes'],
    #                                              q['q_no_answers'], q['q_no_quality_answers']])
    question_features.append(np.hstack([question_feature_matrix[i, :], q['q_no_upvotes'],
                                        q['q_no_answers'], q['q_no_quality_answers']]))

# USER FEATURES

u_column_names = ['u_id', 'e_expert_tags', 'e_desc_word_seq', 'e_desc_char_seq']
word_counts = cv_word.fit_transform(user_dataframe['e_desc_word_seq'].tolist())
tf_word = TfidfTransformer(use_idf=True, norm='l2').fit(word_counts)
word_tfs = tf_word.transform(word_counts)
tsvd = TruncatedSVD(n_components=150, n_iter=7, random_state=42)
word_tfs = tsvd.fit_transform(word_tfs)

char_counts = cv_char.fit_transform(user_dataframe['e_desc_char_seq'].tolist())
tf_char = TfidfTransformer(use_idf=True, norm='l2').fit(char_counts)
char_tfs = tf_char.transform(char_counts)
tsvd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
char_tfs = tsvd.fit_transform(char_tfs)

tag_counts = cv_tag.fit_transform([str(i) for i in user_dataframe['e_expert_tags'].tolist()])
tf_tags = TfidfTransformer(use_idf=True, norm='l2').fit(tag_counts)
tag_tfs = tf_tags.transform(tag_counts)
tsvd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
tag_tfs = tsvd.fit_transform(tag_tfs)

user_feature_matrix = Normalizer(copy=True).fit_transform(np.hstack([word_tfs, char_tfs, tag_tfs]))

user_feature_map = {}
user_features = []
for i, u in user_dataframe.iterrows():
    # user_feature_map[u['u_id']] = np.hstack([user_feature_matrix[i,:]])
    user_features.append(np.hstack([user_feature_matrix[i, :]]))

print 'creating index map...'
q_index_map = {qid: i for i, qid in enumerate(question_dataframe['q_id'].tolist())}
u_index_map = {uid: i for i, uid in enumerate(user_dataframe['u_id'].tolist())}

X = []
y = []
for i, t in train_info_dataframe.iterrows():
    X.append([t['u_id'], t['q_id']])
    y.append(t['answered'])

X = np.array(X)
y = np.array(y)


def print_log(row, header=False, spacing=12):
    top = ''
    middle = ''
    bottom = ''
    for r in row:
        top += '+{}'.format('-' * spacing)
        if isinstance(r, str):
            middle += '| {0:^{1}} '.format(r, spacing - 2)
        elif isinstance(r, int):
            middle += '| {0:^{1}} '.format(r, spacing - 2)
        elif (isinstance(r, float)
              or isinstance(r, np.float32)
              or isinstance(r, np.float64)):
            middle += '| {0:^{1}.5f} '.format(r, spacing - 2)
        bottom += '+{}'.format('=' * spacing)
    top += '+'
    middle += '|'
    bottom += '+'
    if header:
        print(top)
        print(middle)
        print(bottom)
    else:
        print(middle)
        print(top)


def patk_learning_curve(model, train, test,
                        iterarray=None, user_features=None,
                        item_features=None,
                        **fit_params):
    old_epoch = 0
    train_auc = []
    test_auc = []
    headers = ['Epoch', 'train AUC', 'test AUC']
    print_log(headers, header=True)
    for epoch in iterarray:
        more = epoch - old_epoch
        model.fit_partial(train, user_features=user_features,
                          item_features=item_features,
                          epochs=more, **fit_params)
        Xtest_u_ids, Xtest_q_ids = zip(*test)
        Xtest_u_ids = np.array([u_index_map[id] for id in Xtest_u_ids])
        Xtest_q_ids = np.array([q_index_map[id] for id in Xtest_q_ids])
        ypred = model.predict(Xtest_u_ids, Xtest_q_ids,
                              user_features=user_features,
                              item_features=item_features,
                              num_threads=NUM_THREADS)

        this_test = roc_auc_score(ytest, ypred)

        this_train = auc_score(model, train,
                               user_features=user_features,
                               item_features=item_features,
                               num_threads=NUM_THREADS).mean()

        train_auc.append(this_train)
        test_auc.append(this_test)
        row = [epoch, train_auc[-1], test_auc[-1]]
        print_log(row)
    return model, train_auc, test_auc


skf = StratifiedKFold(n_splits=10)
ci = 0
auc = []
user_features = sparse.csr_matrix(user_features, dtype=np.float32),
question_features = sparse.csr_matrix(question_features, dtype=np.float32)
for train_index, test_index in skf.split(X, y):
    ci += 1
    Xtrain, ytrain = X[train_index], y[train_index]
    Xtest, ytest = X[test_index], y[test_index]
    train = np.zeros((len(user_dataframe), len(question_dataframe)), dtype=float)
    user_train_features = []
    user_test_features = []
    question_train_features = []
    question_test_features = []
    iterarray = range(10, 110, 10)
    for i, XV in enumerate(Xtrain):
        train[u_index_map[XV[0]]][q_index_map[XV[1]]] = int(ytrain[i])
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)
    model.fit(train, epochs=0)
    model, train_auc, test_auc = patk_learning_curve(
        model, train, Xtest, iterarray, user_features, question_features,
        **{'num_threads': 24}
    )
    print 'Cross-Val-run:{}\ttrain AUC:{}\ttest AUC:{}'.format(ci, train_auc[-1], test_auc[-1])

