#!/usr/bin/env python3

''' 
Hyperdimensional (HD) binary classifier package using cuda 
'''
import time, sys 
import numpy as np
import torch 
from lsh import lsh
from sklearn.svm import LinearSVC,SVC
from nn_trainer3 import proj_trainer_end_end
from sklearn.svm import LinearSVC

__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "2.5.2018"

class hd_classifier(lsh):

	def __init__(self,feat_dim,spat_dim,HD_dim = 1000, d = 11, encoding = 'single', code = 'thermometer',sparsity = 0.5,learning = 'average',n_classes=4,cuda_device = 'cuda:0',k=0):
		'''	Initialization of HD classifier which inherits from encding locality sensitive hashing class lsh  
		Parameters
		----------
		feat_dim: input feature dimension 
		spat_dim: spatial dimension for spatial encoder (mostly = number of frequency bands ) 
		HD_dim: HD dimension 
		d: number of quantisation bits per feature => only used for 'thermometer' and 'greyish' code 
		encoding:	'single': one single encoding (not divided in spatial dimension)
					'spat'  : spat_dim input feature vectors which are spatialy encoded and added together (with threshold)
					'spat_bind': spatially bind the encoded vectors. 
		code: code used getting binary HD vector by lsh class 
				'thermometer': thermometer code with d bits quantization
				'greyish'    : grey code with 2 bits change between adjacent levels 
				'random_proj_gm': Random projection with Gaussian entries
				'random_proj_bp': Sparse random bipolar (-1,1) random projection 
				'learn_HD_proj' : Learned HD projection 
		sparsity: Share of zeros in random_proj_bp code 
		learning: 'average' : Learning with assotiaive memory by averaging 
				  'SVM'	  : learn one vector per class with SVM and transfom to HD space 
				  'weighted_readout': Weighted Readout of HD components. 
		n_classes: Number of classes 
		'''

		super().__init__(feat_dim,HD_dim,d,code,sparsity,cuda_device=cuda_device) # init lsh for encoding real valued feature vector to binary HD vector 

		self.NO_classes = n_classes
		self.encoding = encoding
		self.spat_dim = spat_dim
		self.learning = learning
		self.code = code
		self.learning = learning

		# Assotiative memory (one vector per class)
		self.ClassItem = torch.ShortTensor(self.NO_classes,self.HD_dim).zero_().to(self.device)
		
		# item memory for bindinig vectors 
		if self.encoding == 'single': # need only one vector in single decoding 
			self.enc_vec = torch.ShortTensor(self.HD_dim).bernoulli_().to(self.device)
			self.transform = self.single_transform # Assign transform function 
		elif self.encoding == 'spat': # need spat_dim vectors
			self.enc_vec = torch.ShortTensor(self.HD_dim,self.spat_dim).bernoulli_().to(self.device) # bernoulli
			self.transform = self.spat_transform 
		elif self.encoding == 'spat_bind': 
			self.enc_vec = torch.ShortTensor(self.HD_dim,self.spat_dim).bernoulli_().to(self.device) # bernoulli
			self.transform = self.spat_bind_transform 
		else:
			raise ValueError('Invalid encoding value. Got ' +self.encoding + ' and expected of spat or single') 

		if learning == 'SVM': 
			self.fit = self.svm_fit 
		elif code == 'learn_HD_proj_SGD': 
			self.fit = self.fit_learn_proj_sgd
		elif code == 'learn_HD_proj_ls':
			self.fit = self.fit_learn_ls
		# elif code == 'learn_HD_proj_unsup':
		# 	self.fit= self.fit_learn_unsup
		else:
			self.fit = self.average_fit
		




		# permutation vector for cyclic shift by one bit 
		self.per = torch.LongTensor(self.HD_dim).to(self.device)
		self.per[1:] =torch.arange(0,self.HD_dim-1)
		self.per[0] = self.HD_dim-1


	def average_fit(self,samples, labels, n_iter = 0):
		'''	Training of HD classifier, perceptron learning
		Parameters
		----------
		samples: feature sample, numpy array shape 	spatial:(NO_samples,temp_dim,spat_dim,N_feat) 
													single:(NO_samples,temp_dim,N_feat)
		lables: label of training data 
		n_iter: number of aditinal training iterations for assotiative memory 

		'''	
		samples,labels = self.flatten_samples(samples,labels)

		# place holder for Assotiative memory 
		ClassItem = torch.ShortTensor(self.NO_classes,self.HD_dim).zero_().to(self.device)
		
		NO_samples = samples.shape[0]
		# counts occurences of class for thresholding 
		class_count = np.zeros(self.NO_classes) 


		#assotiative average learning 
		for i in range(NO_samples):
			#self.get_statistics(samples[i])
			S, cnt = self.transform(samples[i], clipping = False) # get transformed HD_vector(not thresholded yet)
			ClassItem[labels[i]-1].add_(S) 
			class_count[labels[i]-1] += cnt 

		# Thresholding of Assotiative Memory  	
		self.ClassItem = self.thresh_item(ClassItem,self.NO_classes, class_count,self.HD_dim)	# Thresholding of
			
	def svm_fit(self,samples, labels, n_iter = 0):
		'''	Training of HD classifier with SVM strategy
		Parameters
		----------
		samples: feature sample, numpy array shape 	spatial:(NO_samples,temp_dim,spat_dim,N_feat) 
													single:(NO_samples,temp_dim,N_feat)
		lables: label of training data 
		n_iter: number of aditinal training iterations for assotiative memory 

		'''	
		samples,labels = self.flatten_samples(samples,labels)
		
		NO_samples = samples.shape[0]

		svm_train_feat = samples.reshape(NO_samples,-1)
		# train SVM 
		clf = LinearSVC(C = 0.1, intercept_scaling=1, loss='hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=1, tol=0.00001)
		clf.fit(svm_train_feat,labels)
		
		# transform learned vectors to HD 
		for clas in range(self.NO_classes): 
			coef = clf.coef_[clas]
			if self.encoding =='spat' or self.encoding=='spat_bind':
				coef = coef.reshape(self.spat_dim,-1)
			self.ClassItem[clas],_ = self.transform(coef,clipping = True)


	def fit_learn_proj_sgd(self,samples, labels, n_iter = 0):
		'''	Training of HD classifier with SGD end-end
		Parameters
		----------
		samples: feature sample, numpy array shape 	spatial:(NO_samples,temp_dim,spat_dim,N_feat) 
													single:(NO_samples,temp_dim,N_feat)
		lables: label of training data 
		n_iter: number of aditinal training iterations for assotiative memory 

		'''	
		N_epoch = 51
		samples_flat,labels_flat = self.flatten_samples(samples,labels)
		NO_samples, N_bands,N_feat = samples_flat.shape

		end_end_model = proj_trainer_end_end(N_feat,N_bands,self.HD_dim,self.NO_classes,N_epoch,self.device)
		end_end_model.train(samples_flat,labels_flat)

		self.ClassItem, self.proj_mat,self.enc_vec = end_end_model.get_params()

		self.average_fit(samples, labels)

	
	def fit_learn_ls(self,samples,labels,n_iter = 0): 
		'''	Training of HD classifier with LS training 
		Parameters
		----------
		samples: feature sample, numpy array shape 	spatial:(NO_samples,temp_dim,spat_dim,N_feat) 
													single:(NO_samples,temp_dim,N_feat)
		lables: label of training data 
		n_iter: number of aditinal training iterations for assotiative memory 

		'''	
		
		flat_samples,flat_labels = self.flatten_samples(samples,labels)
		N_samples, N_bands,N_feat = flat_samples.shape
		# generate random bit vectors \in {0,1} for every class and frequency band 
		c = np.random.randint(2,size=(self.NO_classes,N_bands,self.HD_dim)) # 
		# stacked data from source space 
		X = np.zeros((N_feat,N_samples*N_bands))
		# stacked values from target space 
		C = np.zeros((self.HD_dim,N_samples*N_bands))

		# fill X with features and C with random vectors 
		for smpl in range(N_samples): 
			# go through all frequency bands 
			for band in range(N_bands):
				# assign corresponding random vector 
				C[:,smpl*N_bands+ band] = c[flat_labels[smpl]-1,band] 
				X[:,smpl*N_bands+ band] = flat_samples[smpl,band]

		# move to GPU 
		target = torch.from_numpy(C).to(self.device)
		# do LS solution 
		C_tilde = 2*C-1
		S = np.matmul(C_tilde,np.linalg.pinv(X))
		self.proj_mat = torch.from_numpy(S).float().to(self.device)

		# train assotiative memory 
		self.average_fit(samples, labels, n_iter = 0)
		



	def flatten_samples(self,samples,labels):

		if self.encoding == 'spat' or self.encoding== 'spat_bind':
			# flatten all temporal windows 
			_,temp_dim,spat_dim,N_feat = samples.shape
			samples = samples.reshape(-1,spat_dim,N_feat)
			labels= labels.repeat(temp_dim)
		elif self.encoding == 'single': 
			# flatten all temporal windows 
			_,temp_dim,N_feat = samples.shape
			samples = samples.reshape(-1,N_feat)
			labels= labels.repeat(temp_dim)
		return samples,labels

	def set_learnable_params(self,ass_mem,proj_mat,enc_vec): 

		self.ClassItem = ass_mem.to(self.device)
		self.proj_mat = proj_mat.to(self.device)
		self.enc_vec = enc_vec.to(self.device)

		return

	
	def predict(self,samples):
		'''	Predict multiple samples
		Parameters
		----------
		samples: feature sample, numpy array shape 	spatial:(NO_samples,1,spat_dim,N_feat) 
													single:(NO_samples,1,N_feat)
		Return  
		------
		y_hat -- Vector of estimated output class, shape (NO_samples)
		HD_score-- Vector of similarities [0,1], shape (NO_samples) 
		'''			
		# use only first temporal window 
		samples = samples[:,0]

		NO_samples = samples.shape[0] 	
		HD_score = np.zeros((NO_samples,self.NO_classes))
		
		y_hat = np.zeros(NO_samples, dtype = int)

		for smpl_idx in range(0,NO_samples):
			# Transform feature vector to binary HD vector
			S , _ = self.transform(samples[smpl_idx],clipping = True) 
			# calculate HD score for every test class 
			for test_class in range(0,self.NO_classes):
				HD_score[smpl_idx,test_class] = 1-self.ham_dist(S,self.ClassItem[test_class]) # get value back in cpu
			# Estimated Class is the one with maximum HD_score (+1 b/c classes [1,2,3,4])
			y_hat[smpl_idx] = np.argmax(HD_score[smpl_idx])+1

		return y_hat, HD_score 

	def single_predict(self,S ):
		
		HD_score = np.zeros(self.NO_classes)
		
		for test_class in range(0,self.NO_classes):
			HD_score[test_class] = 1-self.ham_dist(S,self.ClassItem[test_class]) # get value back in cpu
			# Estimated Class is the one with maximum HD_score (+1 b/c classes [1,2,3,4])
			y_hat = np.argmax(HD_score)+1
		return y_hat,HD_score


	def score(self,samples,labels): 
		'''	Determine accuracy of classification 
		Parameters
		----------
		samples: feature sample, numpy array shape 	spatial:(NO_samples,spat_dim,N_feat) 
													single:(NO_samples,N_feat)
		labels : labels of feature samples, numpy array shape (NO_samples)
		
		Return  
		------
		accuracy: in interval [0,1] 
		'''				
		y_hat,_ = self.predict(samples)

		return 1-np.count_nonzero(y_hat-labels)/labels.size

	def single_transform(self,sample,clipping = True):
		'''	encode real feature vector to binary vector with single encoder 
		Parameters
		----------
		samples: feature sample, numpy array shape (N_feat)
		clipping: not used 
		
		Return  
		------
		bound_vec: binary transformed vector, torch Short Tensor shape(HD_dim)
		cnt: number of additions, (here = 1)
		'''	
		#self.get_statistics(sample)
		hd_vec = self.encode(sample,band=0)
		#if self.code == 'thermometer':
		bound_vec = self.xor(self.enc_vec,hd_vec)
		#else:
		#bound_vec = hd_vec#self.xor(self.enc_vec,hd_vec)
		cnt = 1

		return bound_vec, cnt 

	
	def spat_transform(self,sample, clipping=True): 
		'''	encode real feature vector to binary vector with spatial encoder 
		Parameters
		----------
		samples: feature sample, numpy array shape (N_feat)
		clipping: operation after spatial encoder, True: do thresholding False: no thresholding (has then to be done later)
		
		Return  
		------
		bound_vec: transformed vector, torch Short Tensor shape(HD_dim)
		cnt: number of additions
		'''	
		#self.get_statistics(sample)
		hd_vec = torch.ShortTensor(self.HD_dim).zero_().to(self.device)
		# transform every spatial dimension and add together 
		for item in range(self.spat_dim): 
			hd_vec.add_(self.xor(self.enc_vec[:,item],self.encode(sample[item],band=item)))
		if clipping: 
			# do thresholding 
			return (hd_vec > int(self.spat_dim / 2)).short(), 1 
		else:
			return hd_vec, self.spat_dim  

	def spat_bind_transform(self,sample, clipping=True): 
		'''	encode real feature vector to binary vector with spatial - binding encoder 
		Parameters
		----------
		samples: feature sample, numpy array shape (N_feat)
		clipping: not supported here, allways clipped 
		
		Return  
		------
		bound_vec: transformed vector, torch Short Tensor shape(HD_dim)
		cnt: number of additions
		'''	
		#self.get_statistics(sample)
		hd_vec = torch.ShortTensor(self.HD_dim).zero_().to(self.device)
		# transform every spatial dimension and add together 
		for item in range(self.spat_dim): 
			hd_vec= self.xor(hd_vec,self.xor(self.enc_vec[:,item],self.encode(sample[item],band=item)))
		
		return hd_vec, 1 

	def thresh_item(self,Item,NO_item,item_count,dim,add_dim=0):
		'''	Thresholding of items, if even number we add random vector for breaking ties 
		Parameters
		----------
		Item: accumulated HD vector, torch short tensor shape	if add_dim=0: (NO_item,dim)
																if add_dim!=0:(NO_item,add_dim,dim)
		NO_item: number of items to threshold 
		item_count: number of additions per Item for determining threshold, numpy array shape (NO_item)
		dim : HD dimension 
		add_dim: additional dimension
		Return  
		------
		Item: Thresholded item 
		'''	
		
		for i in range(NO_item): 
			if item_count[i] % 2 == 0: # even number of counts 
				if add_dim == 0: # add a dim- dimensional tensor
					Item[i].add_(torch.ShortTensor(dim).bernoulli_().to(self.device)) # add random vector 
				else: # add a dim x add_dim - dimensional tensor 
					Item[i].add_(torch.ShortTensor(add_dim,dim).bernoulli_().to(self.device)) # add random vector 

				item_count[i] += 1 

			# Tresholding 
			Item[i] = Item[i] > int(item_count[i]/2)

		return Item


	def permute(self, HD_sample): 
		'''	Circular permutation of HD_vector by one bit
		Parameters
		----------
		HD_sample: accumulated HD vector
		Return  
		------
		HD_sample: permuted HD sample 
		'''	
		return HD_sample[self.per]

	def ham_dist(self,vec_a,vec_b):
		''' calculate relative hamming distance 
		Parameters
		----------
		vec_a: first vector, torch Short Tensor shape (HD_dim,)
		vec_b: second vector, torch Short Tensor shape (HD_dim,)
		Return  
		------
		rel_dist: relative hamming distance 
		'''	
		#vec_c = self.xor(vec_a,vec_b)
		rel_dist = 1- torch.sum(vec_a == vec_b).cpu().numpy()/ float(torch.numel(vec_a))
		
		#rel_dist = float(torch.sum(vec_c).cpu().numpy()) / float(torch.numel(vec_c))

		return rel_dist

	def invert(self,vec_a): 
		''' invert binary vector
		Parameters
		----------
		vec_a: input vector, torch Short Tensor shape (HD_dim,)
		Return  
		------
		vec_a*(-1) + 1: inverted vector
		'''	
		return vec_a*(-1) + 1

	def xor(self,vec_a,vec_b):
		''' xor between vec_a and vec_b
		Parameters
		----------
		vec_a: first vector, torch Short Tensor shape (HD_dim,)
		vec_b: second vector, torch Short Tensor shape (HD_dim,)
		Return  
		------
		vec_c: vec_a xor vec_b
		'''	
		vec_c = (torch.add(vec_a,vec_b) == 1).short()  # xor  

		return vec_c

	def get_multi_HD_dist(self,a,dic,n_item = 4):
		# calculate HD dist between c and entries of dic
	
		dist = np.zeros(n_item)

		for i in range(n_item):
			dist[i] = self.ham_dist(a,dic[i])

		return dist 

	def lda_featselect(self,X,y):

		# generate feature indexes 
		clf_LDA = lda_multires(solver='lsqr',shrinkage='auto',precision=64)
		clf_LDA.fit(X,y)