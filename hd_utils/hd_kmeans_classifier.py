#!/usr/bin/env python3

''' 
Hyperdimensional (HD) classifier using k-centroids per class as assotiative memory 
'''
import time, sys 

import numpy as np
import torch 


from lsh import lsh
from sklearn.svm import LinearSVC,SVC

from hd_bin_classifier_cuda import hd_classifier
from HD_Kmeans import KMeans

__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "2.5.2018"

class hd_kmeans_classifier(hd_classifier):
	'''		
	Parameters
	----------
	feat_dim: input feature dimension 
	spat_dim: spatial dimension for spatial encoder (mostly = number of frequency bands ) 
	HD_dim: HD dimension 
	d: number of quantisation bits per feature => only used for 'thermometer' and 'greyish' code 
	encoding:	'single': one single encoding (not divided in spatial dimension)
				'spat'  : spat_dim input feature vectors which are spatialy encoded and added together (with threshold)
	code: code used getting binary HD vector by lsh class 
			'thermometer': thermometer code with d bits quantization
			'greyish'    : grey code with 2 bits change between adjacent levels 
			'random_proj_gm': Random projection with Gaussian entries
			'random_proj_bp': Sparse random bipolar (-1,1) random projection 
			'learn_HD_proj' : Learned HD projection 
	sparsity: Share of zeros in random_proj_bp code 
	learning: 'average' : Learning with assotiaive memory by averaging 
			  'SVM'	  : learn one vector per class with SVM and transfom to HD space 
	n_classes: Number of classes 
	

	'''

	def __init__(self,feat_dim,spat_dim,HD_dim = 1000, d = 11, encoding = 'single', code = 'thermometer',sparsity = 0.5,learning = 'average',n_classes=4,cuda_device = 'cuda:0',k = 1):
		

		super().__init__(feat_dim,spat_dim,HD_dim,d,encoding, code,sparsity,learning,n_classes,cuda_device)

		self.k = k # number of centroids per class 
		self.AssMem = torch.ShortTensor(self.NO_classes,k,self.HD_dim).zero_().to(self.device)
		self.kMean = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=1e-4, cuda_device = cuda_device)
		self.fit = self.kmeans_fit


	def kmeans_fit(self,samples, labels, n_iter = 0):
		'''	Training of HD classifier redefine as K means learning 
		Parameters
		----------
		samples: feature sample, numpy array shape 	spatial:(NO_samples,1,spat_dim,N_feat) 
													single:(NO_samples,1,N_feat)
		lables: label of training data 
		n_iter: number of aditinal training iterations for assotiative memory 

		'''	
		# Train projection if necessairy
		if self.code == 'learn_HD_proj_SGD':
			self.fit_learn_proj_sgd(samples, labels, n_iter = 0)
		elif self.code == 'learn_HD_proj_ls':
			self.fit_learn_proj_ls(samples, labels, n_iter = 0)
		
		# flatten all temporal windows 
		_,temp_dim,spat_dim,N_feat = samples.shape
		samples = samples.reshape(-1,spat_dim,N_feat)
		labels= labels.repeat(temp_dim)
			
		NO_samples = samples.shape[0] 

		# HD vector of transformed vectors 
		ClassItem = torch.ShortTensor(self.NO_classes,NO_samples,self.HD_dim).zero_().to(self.device)
		# counts occurences of class for thresholding 
		class_count = np.zeros(self.NO_classes,dtype = int) 

		# transform every sample and store it in according class row 
		for smpl_idx in range(NO_samples):
			S, _ = self.transform(samples[smpl_idx], clipping = True) # get transformed HD_vector
			label = int(labels[smpl_idx]-1) 
			ClassItem[label,class_count[label]] = S 
			class_count[label] += 1 

		# compute centroids for every 
		for clas in range(self.NO_classes): 
			self.kMean.fit(ClassItem[clas,:class_count[clas]])
			self.AssMem[clas] = self.kMean.cluster_centers_

		return 
	
	def predict(self,samples):
		'''	Predict multiple samples
		Parameters
		----------
		samples: feature sample, numpy array shape 	spatial:(NO_samples,spat_dim,N_feat) 
													single:(NO_samples,N_feat)
		Return  
		------
		y_hat -- Vector of estimated output class, shape (NO_samples)
		HD_score-- Vector of similarities [0,1], shape (NO_samples) 
		'''	
		# use one temporal window 
		samples = samples[:,0]

		NO_samples = samples.shape[0] 	
		
		y_hat = np.zeros(NO_samples, dtype = int)

		HD_score = np.zeros((NO_samples,self.NO_classes))

		for smpl_idx in range(0,NO_samples):

			S, _ = self.transform(samples[smpl_idx], clipping = True) # get transformed HD_vector
			# calculate minimum distance for every class 
			for clas in range(self.NO_classes): 
				score = 1-self.kMean.get_multi_HD_dist(S,self.AssMem[clas],n_item= self.k)
				HD_score[smpl_idx,clas] = np.max(score)

			y_hat[smpl_idx] = np.argmax(HD_score[smpl_idx])+1


		return y_hat, HD_score


	def single_predict(self,S):
		
		HD_score = np.zeros(self.NO_classes)
		
		for clas in range(self.NO_classes): 
			score = 1-self.kMean.get_multi_HD_dist(S,self.AssMem[clas],n_item= self.k)
			HD_score[clas] = np.max(score)

		y_hat = np.argmax(HD_score)+1

		return y_hat,HD_score

	def set_learnable_params(self,ass_mem,proj_mat,enc_vec): 

		self.AssMem = ass_mem.to(self.device)
		self.proj_mat = proj_mat.to(self.device)
		self.enc_vec = enc_vec.to(self.device)

		return
