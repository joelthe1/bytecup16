import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
import pickle

file1_df = pd.read_csv("temp1.csv", names = ['q_id','u_id','label'], sep = ',')
file2_df = pd.read_csv("temp2.csv", names = ['q_id','u_id','label'], sep = ',')

file1_df_sorted = file1_df.sort_values('label', axis=0, ascending=True)
file2_df_sorted = file2_df.sort_values('label', axis=0, ascending=True)

#print file1_df_sorted
#print file2_df_sorted

file1_df_sorted.to_csv(open('temp1.sorted', 'w'))
file2_df_sorted.to_csv(open('temp2.sorted', 'w'))
