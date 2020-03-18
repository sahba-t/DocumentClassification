import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix

sparse_training = lil_matrix((6774, 61189), dtype=np.int16)
sparse_training[:, -1] = np.ones((6774, 1), dtype=np.int16)
with open('../res/testing.csv', 'r') as train_stream:
    for i, line in enumerate(train_stream):
        print(i, len(line))
        current_line = np.array(list(map(int, line.split(','))), dtype=np.int16)
        sparse_training[i, :-1] = current_line[1:]
sparse.save_npz('../res/lr_testing_data', sparse_training.tocsr())
print(sparse_training)
print(sparse_training.shape)
print(sparse_training[2, -1])
print(sparse_training[2, -2])
