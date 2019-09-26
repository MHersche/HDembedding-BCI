#!/usr/bin/env python3

''' 
Run 
'''

import numpy as np
import time, sys 

import sys, os
sys.path.append('../data_utils/')
#sys.path.append('./hd_utils/')
sys.path.append('../baseline_utils/')


from lda_multires import lda_multires
from svm_multires import svm_multires

# import self defined functions 
from load_feature_IV2a import load_feature_IV2a,load_XVAL_feature_IV2a
from load_feature_epfl import load_feature_EPFL


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"



data_path = '../dataset/IV2a/'
NO_subjects = 9
NO_iter = 1 # number of training steps 
n_classes = 4

# svm settings 
svm_kernel = 'linear' #'sigmoid'#'linear' # 'sigmoid', 'rbf',
svm_c = 0.0008 # for linear 0.1 (inverse),	

train_feat = np.zeros((0,10879))
train_label= np.zeros((0,))

test_feat = []
test_label = []

# init SVM for comparison 
clf_SVM = svm_multires(C = svm_c, intercept_scaling=1, loss='hinge', max_iter=1000,
	multi_class='ovr', penalty='l2', random_state=1, tol=0.00001,precision=16)

# training 
for subject in range(1,NO_subjects+1):

	_, tr_feat,tr_label, _, tst_feat,tst_label = load_feature_IV2a(data_path,
			subject, load_features=True)

	train_feat = np.append(train_feat,tr_feat,axis=0)
	train_label = np.append(train_label,tr_label,axis=0)
	test_feat.append(tst_feat)
	test_label.append(tst_label)

# fit 
clf_SVM.fit(train_feat,train_label)

success = np.zeros(NO_subjects)
# test 
for subject in range(NO_subjects):
	success[subject] = clf_SVM.score(test_feat[subject],test_label[subject])
	print("Subject{:}; \t {:0.4f}".format(subject,success[subject]))


mean_suc = np.mean(success)

print("Mean; \t {:0.4f}".format(mean_suc))






	