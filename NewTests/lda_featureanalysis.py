

import sys, os
sys.path.append('../data_utils/')
sys.path.append('../baseline_utils/')

from load_feature_epfl import load_feature_EPFL
import numpy as np
from lda_multires import lda_multires
import matplotlib.pyplot as plt


data_path = '../dataset/3classMI/'


load_feature = load_feature_EPFL
crossval = True # only xval in EPFL dataset 
NO_subjects = 5
NO_folds = 4
NO_iter = 1 # number of training steps 
n_classes = 3
subject = 1
t_vec = np.array([0,4])
f_band=np.arange(13)
feat_type = 'Riemann'
encoding = 'spat'
fold = 1

train_feat, svm_train_feat,train_label, eval_feat, svm_eval_feat,eval_label = load_feature_EPFL(data_path,
			subject, t_vec, f_band,feat_type, encoding, fold,NO_folds,load_features=True)

clf_LDA = lda_multires(solver='lsqr',shrinkage='auto',precision=64)

clf_LDA.fit(svm_train_feat,train_label)

coef = clf_LDA.coef_

[n_clas,n_feat] = coef.shape
coef = coef.reshape((n_clas*n_feat,1))


num_bins = 100
# the histogram of the data
n, bins, patches = plt.hist(coef, num_bins)
plt.show()