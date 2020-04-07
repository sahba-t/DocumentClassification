# Document Classification 
Sahba Tashakkori and Mauricio Monisivais 
## Kaggle Group Name
Sahba: I'll choose the team name next...
## Requirements
* Python 3.6+ (not 3.5 because of 3.6 print statements)
* Pandas, Numpy, Scipy, Scikit Learn, Matplotlib
## Directory Structure
### res
contains resource files, excluding training and testing CSV (they're a couple gigabytes)

### results
contains output files from naive Bayes and logistic regression

### src
#### NaiveBayes.ipynb
The python notebook containing tha naive bayes implementation and code that answers questions related to it.

#### LogisticRegression.ipynb
The python notebook containing tha logistic regression implementation and code that answers questions related to it.

#### make_sparese.py
Makes different sparse_matrices to be used with LR and NB

### plot_lr_ps.py
plots the results of parameter sweeps

### parameter_sweep.py
The code that did actual training and parameter search. The division of training and evaluation data happens here
