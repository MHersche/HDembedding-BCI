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


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "2.5.2018"

class hd_knn(hd_classifier):
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
		self.fit = self.knn_fit
		self.predict = self.knn_predict
		self.k =k 

	def knn_fit(self,samples,labels,n_iter = 0):
		'''	Training of HD classifier with LS training 
		Parameters
		----------
		samples: feature sample, numpy array shape 	spatial:(NO_samples,temp_dim,spat_dim,N_feat) 
													single:(NO_samples,temp_dim,N_feat)
		lables: label of training data 
		n_iter: number of aditinal training iterations for assotiative memory 

		'''	
		
		# do first fitting of SGD if necessary
		if self.code == 'learn_HD_proj_SGD': 
			self.fit_learn_proj_sgd(samples,labels,n_iter)
		elif self.code == 'learn_HD_proj_ls':
			self.fit_learn_ls(samples,labels,n_iter)
		elif self.code == 'learn_HD_proj_unsup':
			self.fit_learn_unsup(samples,labels,n_iter)
		
		samples_flat,labels_flat = self.flatten_samples(samples,labels)
		NO_samples=labels_flat.shape[0]
		S = torch.ShortTensor(NO_samples,self.HD_dim).to(self.device)
		#assotiative average learning 
		for i in range(NO_samples):
			#self.get_statistics(samples[i])
			S[i], _ = self.transform(samples_flat[i], clipping = True) # get transformed HD_vector
            
		self.AM = S
		self.label = labels_flat
		self.nlabel= labels_flat.shape[0]


	
	def knn_predict(self,samples):
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
		y_hat = np.zeros(NO_samples)
		score = np.zeros((NO_samples,self.NO_classes))
		for smpl_idx in range(0,NO_samples):
			Q, _ = self.transform(samples[smpl_idx], clipping = True) # get transformed HD_vector
			dist = self.get_multi_HD_dist(Q,self.AM,self.nlabel)
			y_hat[smpl_idx],score[smpl_idx] = self.knn_class(dist)
            
            
		

		return y_hat,np.array([])
	
	def knn_class(self,dist):
        	'''	Predict multiple samples
		Parameters
		----------
		samples: feature sample, numpy array shape 	spatial:(NO_samples,spat_dim,N_feat) 
													single:(NO_samples,N_feat)
		Return  
		------
		y_hat -- Vector of estimated output class, shape (NO_samples)
		score-- Vector of similarities [0,1], shape (NO_samples) 
		'''	
        	sim = 1-dist
        
        	max_idx = np.argpartition(sim,-self.k)[-self.k:]
        
        	max_sim = sim[max_idx]
        	max_label = self.label[max_idx]
        	score = np.zeros(self.NO_classes) 
        	for clas in range(self.NO_classes): 
            		clas_idx = max_label == (clas+1)
            		score[clas] = np.sum(np.exp(max_sim[clas_idx]))
            
        	score = score/np.sum(score)
        	y_hat = np.argmax(score)+1
        	return y_hat, score 
        
    

	def set_learnable_params(self,ass_mem,proj_mat,enc_vec): 

		self.proj_mat = proj_mat.to(self.device)
		self.enc_vec = enc_vec.to(self.device)

		return

