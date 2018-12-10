#!/usr/bin/env python3

''' 
Kmeans clustering 
'''
import time, sys 

import numpy as np
from hd_bin_classifier_cuda import hd_classifier
import torch


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "27.5.2018"

class KMeans:

	def __init__(self, n_clusters=8, init='k-means++', n_init=10,
				 max_iter=300, tol=1e-4, precompute_distances='auto',
				 verbose=0, random_state=None, copy_x=True,
				 n_jobs=1, algorithm='auto', 
				 cuda_device = 'cuda:0'):
		self.n_clusters = n_clusters
		self.init = init
		self.max_iter = max_iter
		self.tol = tol
		self.precompute_distances = precompute_distances
		self.n_init = n_init
		self.verbose = verbose
		self.random_state = random_state
		self.copy_x = copy_x
		self.n_jobs = n_jobs
		self.algorithm = algorithm
		self._best_cost = 0.

		use_cuda = torch.cuda.is_available() 
		self.device = torch.device(cuda_device if use_cuda else "cpu")
	

	def fit(self, X):
		"""Compute k-means clustering.

		Parameters
		----------
		X : array-like or sparse matrix, shape=(n_samples, n_features)
			Training instances to cluster.

		"""
		'''
		calculate K-means algorithm based on HD arithmetic 
		'''
		n_samples, HD_dim = X.shape
		
		if self.verbose: 
				print('Start K-menas calculation for {:d} centroids'.format(self.n_clusters))

		for i_init in range(self.n_init): 

			cur_centroids = self._init_centroids(X,self.n_clusters)
			

			label = np.ones(n_samples,dtype = int)
			new_label = np.zeros(n_samples,dtype = int)

			iterr = 0
			while (not np.array_equal(new_label,label)) and (iterr < self.max_iter):
				# init new centroid for adding up all means 
				new_centroids = torch.ShortTensor(self.n_clusters,HD_dim).zero_().to(self.device)
				class_cnt = np.zeros(self.n_clusters)
				cost = 0

				label = new_label.copy()
				
				# calculate distance and assign labels to closes point
				for samp_idx in range(n_samples): 
					# assign label to closest point
					dist = self.get_multi_HD_dist(X[samp_idx],cur_centroids,self.n_clusters)
					new_label[samp_idx] = np.argmin(dist)
					# update cost 
					cost +=  dist[new_label[samp_idx]]
					# add to new mean
					new_centroids[new_label[samp_idx]].add_(X[samp_idx])
					class_cnt[new_label[samp_idx]] += 1 

				# thresholding of new mean 
				cur_centroids = self.thresh_item(new_centroids,class_cnt)	

				iterr +=1 

			

			if (i_init ==0) or (cost < self._best_cost): # first round or new best partition 
				self._best_cost = cost
				self.cluster_centers_ = cur_centroids
				self.labels_ = new_label

			if self.verbose: 
				print('Iteration {}, Cost: {:.2f}, Best cost:  {:.2f}: '.format(i_init,cost,self._best_cost))

			self._is_fitted = True

		return self

	def fit_predict(self, X, y=None):
		"""Compute cluster centers and predict cluster index for each sample.

		Convenience method; equivalent to calling fit(X) followed by
		predict(X).

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			New data to transform.

		u : Ignored

		Returns
		-------
		labels : array, shape [n_samples,]
			Index of the cluster each sample belongs to.
		"""

		return self.fit(X).labels_

	def predict(self, X):
		"""Predict the closest cluster each sample in X belongs to.

		In the vector quantization literature, `cluster_centers_` is called
		the code book and each value returned by `predict` is the index of
		the closest code in the code book.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			New data to predict.

		Returns
		-------
		labels : array, shape [n_samples,]
			Index of the cluster each sample belongs to.
		"""
		
		if not self._is_fitted:
			raise ValueError("Centroids not fitted ")

		return self._labels_inertia(X)[0]



	def labels_inertia(self,X):


		if not self._is_fitted:
			raise ValueError("Centroids not fitted ")

		n_samples = X.shape[0]

		labels = np.zeros(n_samples,dtype = int) # esstimated labels 
		dist = np.zeros((n_samples,self.n_clusters))

		label_cnt = np.zeros(self.n_clusters,dtype = int)
		labeld_dist= np.zeros((self.n_clusters,n_samples))


		for samp_idx in range(n_samples):
			dist[samp_idx] = self.get_multi_HD_dist(X[samp_idx],self.cluster_centers_,n_item= self.n_clusters)
			label =  np.argmin(dist[samp_idx])
			labels[samp_idx]= label
			# store best label and distance for statistics 
			labeld_dist[label,label_cnt[label]] = np.min(dist[samp_idx])
			label_cnt[label] +=1

		self.var_ = np.zeros(self.n_clusters)
		self.mean_ = np.zeros(self.n_clusters)

		for label in range(self.n_clusters): 
			self.var_[label]= np.var(labeld_dist[label,:label_cnt[label]])
			self.mean_[label]= np.mean(labeld_dist[label,:label_cnt[label]])

		return labels,dist

	def _init_centroids(self,X, k, init= 'k-means++', random_state=None):
		"""Compute the initial centroids

		Parameters
		----------

		X : array, shape (n_samples, n_features)

		k : int
			number of centroids

		init : {'k-means++', 'random' or ndarray or callable} optional
			Method for initialization

		random_state : int, RandomState instance or None, optional, default: None
			If int, random_state is the seed used by the random number generator;
			If RandomState instance, random_state is the random number generator;
			If None, the random number generator is the RandomState instance used
			by `np.random`.

		Returns
		-------
		centers : array, shape(k, n_features)
		"""
		n_samples = X.shape[0]

		init_indices = np.random.choice(n_samples,k,replace = False)
		
		centers = X[init_indices]

		return centers

	def get_multi_HD_dist(self,test_vec,dic,n_item = 4):
		# calculate HD dist between c and entries of dic
		

		n_item = dic.shape[0]

		dist = np.zeros(n_item)

		for i in range(n_item):
			dist[i] = self.ham_dist(test_vec,dic[i])

		return dist 


	def thresh_item(self,Item,item_count):
		'''	Thresholding of items, if even number we add random vector for breaking ties 
		Parameters
		----------
		Item: accumulated HD vector, torch short tensor shape=(NO_item,HD_dim)
		item_count: number of additions per Item for determining threshold, numpy array shape (NO_item)
		Return  
		------
		Item: Thresholded item 
		'''	

		NO_item,HD_dim = Item.shape

		for i in range(NO_item): 
			if item_count[i] % 2 == 0: # even number of counts 
				Item[i].add_(torch.ShortTensor(HD_dim).bernoulli_().to(self.device)) # add random vector 
				item_count[i] += 1 

			# Tresholding 
			Item[i] = Item[i] > int(item_count[i]/2)

		return Item

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
		vec_c = self.xor(vec_a,vec_b)

		rel_dist = float(torch.sum(vec_c).cpu().numpy()) / float(torch.numel(vec_c))

		return rel_dist

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

	def HDmean(X):
		''' HD mean of X 
		Parameters
		----------
		X: HD vectors, torch Short Tensor shape (N_samples,HD_dim,)
		Return  
		------
		out: Mean HD vector, torch Short Tensor shape (HD_dim,)
		'''	
		n_samples,HD_dim = X.shape

		summ = X[0]

		for samp in range(1,n_samples): 
			summ.add_(X[samp])

		out = summ > int(n_samples/2)

		return out 