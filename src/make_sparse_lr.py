import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix


def make_sparse_training(for_bayes=False):
    sparse_training = lil_matrix((12000, 61189), dtype=np.int16)
    if not for_bayes:
        sparse_training[:, -1] = np.ones((12000, 1), dtype=np.int16)

    sparse_testing = lil_matrix
    with open('../res/training.csv', 'r') as train_stream:
        for i, line in enumerate(train_stream):
            current_line = np.array(list(map(int, line.split(','))), dtype=np.int16)
            if for_bayes:
                sparse_training[i] = current_line[1:]
            else:
                sparse_training[i, :-1] = current_line[1:-1]
    out_path = '../res/' + 'nb' if for_bayes else 'lr' + '_training_data'
    sparse.save_npz(out_path, sparse_training.tocsr())
    print(sparse_training)
    print(sparse_training.shape)
    print(sparse_training[2, -1])
    print(sparse_training[2, -2])


def make_sparse_testing(for_bayes=False):
    sparse_testing = lil_matrix((6774, 61188 if for_bayes else 61187), dtype=np.int16)
    if for_bayes:
        sparse_testing[:, -1] = np.ones((6774, 1), dtype=np.int16)

    with open('../res/testing.csv', 'r') as train_stream:
        for i, line in enumerate(train_stream):
            current_line = np.array(list(map(int, line.split(','))), dtype=np.int16)
            if for_bayes:
                sparse_testing[i] = current_line[1:]
            else:
                sparse_testing[i, :-1] = current_line[1:]
    out_path = '../res/' + ('nb' if for_bayes else 'lr') + '_testing_data'
    sparse.save_npz(out_path, sparse_testing.tocsr())
    print(sparse_testing)
    print(sparse_testing.shape)
    print(sparse_testing[2, -1])


# make_sparse_training(for_bayes=True)
make_sparse_testing(for_bayes=True)
