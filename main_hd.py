#!/usr/bin/env python3

''' 
Run 
'''

import numpy as np
import time, sys 

import sys, os
sys.path.append('./data_utils/')
sys.path.append('./hd_utils/')

from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import KFold


# import self defined functions 
from load_feature_IV2a import load_feature_IV2a,load_XVAL_feature_IV2a
from load_feature_epfl import load_feature_EPFL

from hd_bin_classifier_cuda import hd_classifier
from hd_kmeans_classifier import hd_kmeans_classifier

import torch 



__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"

class Hd_model:

	def __init__(self,data_set,data_path,crossval = False,classifier = 'assotiative',cuda_device = 'cuda:0'):		
		
		if data_set =="EPFL":
			self.load_feature = load_feature_EPFL
			self.crossval = True # only xval in EPFL dataset 
			self.NO_subjects = 5
			self.NO_folds = 4
			self.NO_iter = 1 # number of training steps 
			self.n_classes = 3

		elif data_set == "IV2a": 
			if crossval:
				self.load_feature = load_XVAL_feature_IV2a
				self.NO_folds = 4
			else: 
				self.load_feature = load_feature_IV2a
				self.NO_folds = 1

			self.crossval = crossval
			self.NO_subjects = 9
			self.NO_iter = 1 # number of training steps 
			self.n_classes = 4

		self.cuda_device = cuda_device	
		self.data_set = data_set
		self.data_path= data_path
		self.save_path = '' # save_path has to be set 

		# svm settings 
		self.svm_kernel = 'linear' #'sigmoid'#'linear' # 'sigmoid', 'rbf',
		self.svm_c = 0.05 # for linear 0.1 (inverse),	

		# HD classifier settings 
		if classifier == 'assotiative':
			self.hd_classifier = hd_classifier 
		elif classifier == 'kmeans':
			self.hd_classifier = hd_kmeans_classifier
		self.classifier = classifier
	
		
		################## Default settings ##################################
		# feature settings 
		self.feat_type = 'Riemann'
		self.t_vec = [0]
		self.f_band = [0]
		
		# HD settings
		self.k = 1 # number of centroids 
		self.HD_dim = 1000
		self.N_feat_per_band = 1000
		self.sparsity = 0.5
		self.N_bands = 0
		self.d = 7
		self.code = 'thermometer'
		self.encoding = 'spat'
		self.learning = 'average'


	def test_hd(self,fold = 0):
		# load features 
		train_feat, svm_train_feat,train_label, eval_feat, svm_eval_feat,eval_label = self.load_feature(self.data_path,
			self.subject, self.t_vec, self.f_band,self.feat_type, self.encoding, fold,self.NO_folds)

		# init SVM for comparison 
		clf = LinearSVC(C = self.svm_c, 
			intercept_scaling=1, loss='hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=1, tol=0.00001)

		# init HD classifier  
		hd_b = self.hd_classifier(self.N_feat_per_band , 
			self.N_bands,self.HD_dim,self.d,self.encoding, 
			self.code, self.sparsity, self.learning,self.n_classes,self.cuda_device,self.k)


		suc =  np.zeros(3)
		
		################################# Training ###################################################
		# SVM 
		clf.fit(svm_train_feat,train_label)
		# HD
		hd_b.fit(train_feat,train_label)
	
		################################# Evaluation ###################################################
		# success on svm 
		suc[0] = clf.score(svm_eval_feat,eval_label)

		# HD success on all eval data 
		suc[1,] =hd_b.score(eval_feat,eval_label)

		# HD success on training data 
		suc[2] =hd_b.score(train_feat,train_label)

		del hd_b
		torch.cuda.empty_cache()

		return suc
		

	def run(self):

		print(self.data_set)
		print(self.classifier + str(self.k))
		
		self.N_bands = len(self.f_band)

		print(self.feat_type)
		print("Number of frequency bands " + str(self.N_bands))
		print("Number of used features per band: "+ str(self.N_feat_per_band))
		print("Total number of features: " + str(self.N_bands*self.N_feat_per_band))
		
		print(self.code)
		print(self.encoding)
		
		if self.encoding == 'random_proj_bp':
			print("Sparsity: " + str(self.sparsity))
		elif self.encoding == 'thermometer' or self.encoding == 'greyish': 
			print("d = " + str(self.d))
			self.HD_dim = self.N_feat_per_band*self.d

		print("HD dimension: " + str(self.HD_dim))
		print("Learning with : "+ self.learning)
		

		success = np.zeros((self.NO_subjects,self.NO_folds,3))

		print("\t SVM\tHD\tHD training")
		# Go through all subjects 
		for self.subject in range(1,self.NO_subjects+1):
			
			for fold in range(self.NO_folds): 
				success[self.subject-1,fold] = self.test_hd(fold)

			mean_suc = np.mean(success[self.subject-1],axis = 0)

			print("Subject{:} {:0.4f}\t{:0.4f}\t{:0.4f}".format(self.subject,mean_suc[0],mean_suc[1],mean_suc[2]))

		# calculate average accuracies 
		avg_succ = success.mean(axis = (0,1))

		print("AVG: \t {:0.4f} \t {:0.4f} \t {:0.4f}".format(mean_suc[0],mean_suc[1],mean_suc[2]))

		# save data 
		np.savez(self.save_path,success = success,N_feat_per_band = self.N_feat_per_band,
			HD_dim = self.HD_dim, d = self.d, code = self.code,
			sparsity = self.sparsity,data_set = self.data_set,classifier = self.classifier,
			k = self.k)
