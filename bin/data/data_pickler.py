import pandas as pd

table = pd.read_csv('user_info.txt', sep='\t', header=None, names=['expert_id', 'domain_of_expertise', 'desc_in_words', 'desc_in_chars'])
table.to_pickle('user_info.pkl')

table = pd.read_csv('question_info.txt', sep='\t', header=None, names=['ques_id', 'domain_of_question', 'desc_in_words', 'desc_in_chars', 'num_upvotes', 'num_answers', 'num_top_quality_answers'])
table.to_pickle('question_info.pkl')

table = pd.read_csv('invited_info_train.txt', sep='\t', header=None, names=['ques_id', 'expert_id', 'answered_bool'])
table.to_pickle('invited_info_train.pkl')

table = pd.read_csv('validate_nolabel.txt', header=None, names=['ques_id', 'expert_id'])
table.to_pickle('validate_nolabel.pkl')

