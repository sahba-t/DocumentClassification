import time
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix
import datetime
# the number of features
n = 61188
#number of classes
K = 20
training_labels = np.load('../res/lr_labels.npy')

def train_and_eval(etha, lambd, iters, delta, X, XT, file_ptr):
    weights_matrix = np.random.rand(K, n + 1)
    # weights_sparse is the W matrix in the pdf
    weights_sparse = sparse.csr_matrix(weights_matrix, dtype=np.float)
    # do training
    t0 = time.time()
    for i in range(iters):
        pxy = weights_sparse * XT
        #normalize before exp
        pxy = pxy.multiply(1/pxy.sum(axis=0)).tocsr()
        pxy = pxy.expm1().tocsr()
        #normalize after exp to get probabilities
        pxy = pxy.multiply(1/pxy.sum(axis=0)).tocsr()
        diff_exp = delta - pxy
        new_w = weights_sparse + etha * (diff_exp * X - weights_sparse.multiply(lambd))
        weights_sparse = new_w
    t1 = time.time()    
    predictions = (weights_sparse * XT)
    predictions = predictions.multiply(1/predictions.sum(axis=0))
    predictions = predictions.expm1()
    predictions = predictions.multiply(1/predictions.sum(axis=0))
    predictions = predictions.todense()
    correct = 0
    for i in range(predictions.shape[1]):
        predicted = np.argmax(predictions[:,i]) + 1
        actual = training_labels[i]
        if predicted == actual:
            correct += 1
    # etha, lambda, accuracy, iterations, time
    file_ptr.write(F"{etha},{lambd},{correct * 100/predictions.shape[1]},{iters},{t1 - t0}\n")
    file_ptr.flush()



def sweep_parametes():
    file_name = str(datetime.datetime.now()) + ".json"
    iterations = 1000
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
    with open(file_name, 'w') as out_stream:
        out_stream.write(F"etha,lambda,accuracy,iterations,time\n")
        ethas = np.arange(0.001,0.01,0.001)
        lambds = np.arange(0.001,0.01,0.001)    
        for etha in ethas:
            for lambd in lambds:
                train_and_eval(etha=0.005, lambd=0.005, iters=iterations, delta=delta, X=training_data_sparse,\
                    XT=XT, file_ptr=out_stream)





if __name__ == "__main__":
    sweep_parametes()