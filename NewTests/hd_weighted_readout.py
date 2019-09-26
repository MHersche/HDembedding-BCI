#!/usr/bin/env python3

''' 
Hyperdimensional (HD) classifier using k-centroids per class as assotiative memory 
'''
import time, sys 

import numpy as np
import torch 


from lsh import lsh
sys.path.append('../baseline_utils/')
from svm_multires import svm_multires

from hd_bin_classifier_cuda import hd_classifier


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "2.5.2018"

class hd_weighted_readout(hd_classifier):
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
	!!!!!!!!!!!!!!!!!!!!!!!!!
	k here is the number of levels: 0 is not quantized 
									2 is bipolar 
									
	

	'''

	def __init__(self,feat_dim,spat_dim,HD_dim = 1000, d = 11, encoding = 'single', code = 'thermometer',sparsity = 0.5,learning = 'average',n_classes=4,cuda_device = 'cuda:0',k = 1):
		

		super().__init__(feat_dim,spat_dim,HD_dim,d,encoding, code,sparsity,learning,n_classes,cuda_device)
		self.fit = self.weighted_fit
		self.k = k

	def weighted_fit(self,samples,labels,n_iter = 0):
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
		
		samples_flat,labels_flat = self.flatten_samples(samples,labels)

		NO_samples=labels_flat.shape[0]
		S = torch.Tensor(NO_samples,self.HD_dim).to(self.device)
		#assotiative average learning 
		for i in range(NO_samples):
			#self.get_statistics(samples[i])
			S[i], _ = self.transform(samples_flat[i], clipping = True) # get transformed HD_vector

		Snp = S.cpu().numpy()*2-1
		# train SVM on HD vectors 
		self.svm_clf = svm_multires(C = 0.1, intercept_scaling=1, loss='hinge', max_iter=1000,
			multi_class='ovr', penalty='l2', random_state=1, tol=0.00001,precision=self.k)
		self.svm_clf.fit(Snp,labels)


	
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
		
		S = torch.FloatTensor(NO_samples,self.HD_dim).to(self.device)

		for smpl_idx in range(0,NO_samples):
			S[smpl_idx], _ = self.transform(samples[smpl_idx], clipping = True) # get transformed HD_vector
			
		Snp = S.cpu().numpy()*2-1
		y_hat = self.svm_clf.predict(Snp)

		return y_hat,np.array([])
			

	def set_learnable_params(self,ass_mem,proj_mat,enc_vec): 

		self.proj_mat = proj_mat.to(self.device)
		self.enc_vec = enc_vec.to(self.device)

		return
