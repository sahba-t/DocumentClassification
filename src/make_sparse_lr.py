import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix

sparse_training = lil_matrix((12000, 61188), dtype=np.int16)
labels = np.zeros(12000, dtype=np.int8)
i = 0
with open('training.csv', 'r') as train_stream:
    for line in train_stream:
        current_line = np.array(list(map(int, line.split(','))), dtype=np.int16)
        labels[i] = current_line[-1]
        sparse_training[i,:] = current_line[1:-1]
        i += 1
        print(i)
sparse.save_npz('lr_training_data', sparse_training.tocsr())