import time
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix
import datetime
import sys
import math
# the number of features
n = 61188
#number of classes
K = 20
training_labels = np.load('../res/lr_labels.npy')

def train_and_eval(etha, lambd, iters, delta, X, train_ratio, file_ptr):
    rows  = X.shape[0]
    last_train_index = math.ceil(train_ratio * rows)
    X_train = X[:last_train_index, :]
    XT_train = X_train.transpose()
    delta_train = delta[:, :last_train_index]
    
    if train_ratio == 1:
        XT_eval = XT
        delta_eval = delta
        last_train_index = 0
    else:
        XT_eval = X[last_train_index:, :].transpose()
        delta_eval = delta[:, last_train_index:]
    
    weights_matrix = np.random.rand(K, n + 1)
    # weights_sparse is the W matrix in the pdf
    weights_sparse = sparse.csr_matrix(weights_matrix, dtype=np.float)
    # do training
    t0 = time.time()
    for i in range(iters):
        pxy = weights_sparse * XT_train
        #normalize before exp
        pxy = pxy.multiply(1/pxy.sum(axis=0)).tocsr()
        pxy = pxy.expm1().tocsr()
        #normalize after exp to get probabilities
        pxy = pxy.multiply(1/pxy.sum(axis=0)).tocsr()
        diff_exp = delta_train - pxy
        new_w = weights_sparse + etha * (diff_exp * X_train - weights_sparse.multiply(lambd))
        weights_sparse = new_w
    t1 = time.time()    
    predictions = (weights_sparse * XT_eval)
    predictions = predictions.multiply(1/predictions.sum(axis=0))
    predictions = predictions.expm1()
    predictions = predictions.multiply(1/predictions.sum(axis=0))
    predictions = predictions.todense()
    correct = 0
    for i in range(predictions.shape[1]):
        predicted = np.argmax(predictions[:, i]) + 1
        actual = training_labels[i + last_train_index]
        if predicted == actual:
            correct += 1
    
    output_str = "%.3f,%.3f,%.4f,%d,%.2f\n" %(etha, lambd, correct * 100/predictions.shape[1], iters, t1 - t0)
    if file_ptr:
        # etha, lambda, accuracy, iterations, time
        file_ptr.write(output_str)
        file_ptr.flush()
    else:
        print(output_str)
        sys.stdout.flush()
    #saving the weights
    matrix_file_name = "../res/weights_{}_{}_{}".format(etha, lambd, iters)
    sparse.save_npz(matrix_file_name, weights_sparse)


def sweep_parametes():
    file_name = "1000_runs" + ".json"
    iterations = 10000
    #number of classes
    #training instances
    M = 12000
    Normalize_X = False

    #loading the normalized training data!
    training_data_sparse = sparse.load_npz('../res/lr_training_data.npz')
    if Normalize_X:
        training_data_sparse = training_data_sparse.multiply(training_data_sparse.sum(axis=0)).tocsr()
    
    delta = lil_matrix(np.zeros((K, M)), dtype = np.int16)
    for i, label in enumerate(training_labels):
        delta[label - 1, i] = 1
    delta = delta.tocsr()
    with open(file_name, 'w') as out_stream:
        out_stream.write(F"etha,lambda,accuracy,iterations,time\n")
        ethas = np.arange(0.008,0.0101,0.001)
        lambds = [0.001, 0.003,0.004,0.005, 0.006,0.008, 0.009, 0.01]
        for etha in ethas:
            for lambd in lambds:
                train_and_eval(etha=etha, lambd=lambd, iters=iterations, delta=delta, X=training_data_sparse,\
                train_ratio=1, file_ptr=out_stream)


def try_single_param():
    train_ratio = 0.85
    eta = float(sys.argv[1])
    lambd = float(sys.argv[2])
    iterations = 10000
    #number of classes
    #training instances
    M = 12000
    Normalize_X = False

    #loading the normalized training data!
    training_data_sparse = sparse.load_npz('../res/lr_training_data.npz')
    if Normalize_X:
        training_data_sparse = training_data_sparse.multiply(training_data_sparse.sum(axis=0)).tocsr()
    
    XT = training_data_sparse.transpose()
    
    delta = lil_matrix(np.zeros((K, M)), dtype = np.int16)
    for i, label in enumerate(training_labels):
        delta[label - 1, i] = 1
    delta = delta.tocsr()
    train_and_eval(etha=eta, lambd=lambd, iters=iterations, delta=delta, X=training_data_sparse,\
        train_ratio=train_ratio, file_ptr=None)

if __name__ == "__main__":
    # sweep_parametes()
    try_single_param()