import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix



def make_sparse_training():
    sparse_training = lil_matrix((12000, 61189), dtype=np.int16)
    sparse_training[:, -1] = np.ones((6774, 1), dtype=np.int16)

    sparse_testing = lil_matrix
    with open('../res/training.csv', 'r') as train_stream:
        for i, line in enumerate(train_stream):
            current_line = np.array(list(map(int, line.split(','))), dtype=np.int16)
            sparse_training[i, 1:-1] = current_line[1:-1]
    sparse.save_npz('../res/lr_training_data', sparse_training.tocsr())
    print(sparse_training)
    print(sparse_training.shape)
    print(sparse_training[2, -1])
    print(sparse_training[2, -2])


def make_sparse_testing():

    sparse_testing = lil_matrix((6774, 61189), dtype=np.int16)
    sparse_testing[:, -1] = np.ones((6774, 1), dtype=np.int16)

    with open('../res/testing.csv', 'r') as train_stream:
        for i, line in enumerate(train_stream):
            current_line = np.array(list(map(int, line.split(','))), dtype=np.int16)
            sparse_testing[i, :-1] = current_line[1:]
    sparse.save_npz('../res/lr_testing_data', sparse_testing.tocsr())
    print(sparse_testing)
    print(sparse_testing.shape)
    print(sparse_testing[2, -1])

make_sparse_testing()