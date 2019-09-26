#!/usr/bin/env python3

''' 
Local sensitive Hashing function for binarization using different encoding strategies
- Thermometer encoding 
- Gray encoding 
- Gaussian random projection 
- Bipolar sparse random projection 
- Learned projections 
'''
import time, sys 
import numpy as np
import scipy.special

# plots 
import matplotlib.pyplot as plt
import torch 
from sklearn.random_projection import SparseRandomProjection 
from sklearn.cluster import KMeans  

__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"

class lsh:

	def __init__(self, NO_feat,HD_dim, d, code,sparsity = 0.9,cuda_device = 'cuda:0'):
		self.training_done = False 
		self.sigma = 1
		self.mean = 0 
		self.code = code 
		self.NO_feat = NO_feat
		
		self.cuda_device = cuda_device
		use_cuda = torch.cuda.is_available() 
		self.device = torch.device(cuda_device if use_cuda else "cpu")

		if code == 'thermometer':
			self.HD_dim = d*NO_feat # HD dimension depends on number of quantization bits 
			self.d = d # number bits for quantization 
			self.dict = torch.from_numpy(np.tril(np.ones(self.d, dtype = int))).short().to(self.device)
			self.q = d # number of quantization levels equals to bits  
			self.encode = self.encode_quant
			
		elif code == 'greyish':
			self.HD_dim = d*NO_feat
			self.d = d # number bits for quantization 
			self.q = int(scipy.special.binom(d,2))
			self.get_greyish_code(d) # determine grey dictionary 
			self.encode = self.encode_quant

		elif code == 'random_proj_gm':
			self.HD_dim = HD_dim
			self.proj_mat = torch.FloatTensor(self.HD_dim, self.NO_feat).normal_().to(self.device)
			self.encode = self.encode_proj
		
		elif code == 'random_proj_bp':
			self.encode = self.encode_proj
			self.HD_dim = HD_dim

			proj_mat_cpu = SparseRandomProjection(HD_dim,density=1-sparsity).fit(np.zeros((1,NO_feat))).components_.todense()
			proj_mat_cpu[proj_mat_cpu >0] = 1
			proj_mat_cpu[proj_mat_cpu <0] = -1
			actual_sparsity = (proj_mat_cpu ==0).mean()
			self.proj_mat = torch.from_numpy(proj_mat_cpu).float().to(self.device)
		
			#print("Actual Sparsity: " + str(actual_sparsity))

		elif code == 'learn_HD_proj_ls' or code == 'learn_HD_proj_SGD' or code == 'learn_HD_proj_unsup': 
			# projection matrix is learned later on during training 
			self.HD_dim = HD_dim 
			self.encode = self.encode_proj
			
		else:
			raise ValueError("No valid encoding! got "+ code)
			
		# create permutation vecor 
		vec = np.arange(self.HD_dim)
		self.perm_vec = np.random.permutation(vec)
		

	def get_greyish_code(self,d):
		'''	generate grey like code: two bits change per between two adjacient quantization levels 
		Parameters: 
		----------
		d: number of bits per value
		
		'''	
		seed = np.zeros(d).astype(bool) # initial value (first level)
		# dictionary 
		self.dict = np.zeros((self.q+1,d)).astype(bool)
		self.dict[0] = seed 

		row = 1

		for ii in range(self.q):
			for jj in range(ii+1,d):
				self.dict[row] = self.dict[row-1]
				change_vec = [ii,jj]
				self.dict[row,change_vec] = np.invert(self.dict[row-1,change_vec])
				row +=1
		self.dict= torch.from_numpy(self.dict[:self.q].astype(int)).short().to(self.device)

	def get_statistics(self, samples):
		'''	save statistics of training data for later standartiztion 
		'''	
		#if ~self.training_done:
		self.sigma = np.std(samples)
		self.mean = np.mean(samples)
		self.training_done = True
		#return 

	def encode_quant(self, sample):	
		'''	feature by quantization (thermometer or grey)
		Parameters: 
		----------
		sample: feature sample, numpy array shape (N_feat,) 
		
		Return:  
		-------
		out_vec: binary HD sample, torch Tensor int16 (HD_dim,)
		'''	
		N_feat = sample.size

		hd_vec = np.zeros(N_feat*self.d)
		
		# standardisation of this sample 
		self.mean = np.mean(sample)
		if self.code == 'thermometer': 
			self.sigma = np.std(sample)*3
		else: 
			self.sigma = np.std(sample)

		sample = (((sample - self.mean)/(self.sigma))*self.q + (self.q-1)/2).astype(int) # add q term for quantization 

		# quantization 
		sample[sample > (self.q-1)] = (self.q-1)
		sample[sample < 0] = 0
		
		out_vec = self.dict[sample].reshape(N_feat*self.d)

		return out_vec
		

	def encode_proj(self,sample): 
		'''	feature transformation by binary projection (random or learned)
		Parameters: 
		----------
		sample: feature sample, numpy array shape (N_feat,) 
		
		Return:  
		----------
		out_vec: binary HD sample, torch Tensor int16 (HD_dim,)
		'''	
		# move sample to GPU 
		sigma = np.std(sample)
		mean = np.mean(sample)

		sample = (sample-mean)/sigma

		#import pdb
		#pdb.set_trace()

		cuda_sample = torch.from_numpy(sample).float().to(self.device)
		# projection 
		proj_vec = torch.matmul(self.proj_mat,cuda_sample)
		# binariyation based on sign 
		out_vec = (proj_vec >= 0).short() 

		return out_vec

	def encode_multi(self, samples): 
		'''	encode multiple sampeles
		Parameters: 
		----------
		samples: feature samples, numpy array shape (N_samples,N_feat) 
		
		Return:  
		----------
		out_vec: binary HD samples, torch Tensor int16 (N_samples,HD_dim)
		'''	
		N_samples, _ = samples.shape

		hd_vec = np.zeros((N_samples,self.HD_dim))

		for samp_idx in range(N_samples):
			hd_vec[samp_idx] = self.encode(samples[samp_idx])

		# move to GPU (maybe already done but does not hurt)
		out_vec = torch.from_numpy(hd_vec).short().to(self.device)

		return out_vec

	def save_model_data(self,subject,fold,path):
		
		# save projection matrix 
		proj_mat_numpy = self.proj_mat.cpu().numpy()
		save_path = path + 'S' + str(subject) + '_' + str(fold) + '.npy'
		np.savez(save_path, proj_mat = proj_mat_numpy)


	def load_model_data(self,subject,fold,path):
		
		# save projection matrix 
		load_path = path + 'S' + str(subject) + '_' + str(fold) + '.npy.npz'

		with np.load(load_path) as data: 
			proj_mat_numpy = data['proj_mat']

			HD_dim,NO_feat = proj_mat_numpy.shape

			if not(NO_feat == self.NO_feat and HD_dim == self.HD_dim): 
				raise ValueError("Number of features and HD dim don't match with loaded lsh mat")

		self.proj_mat = torch.from_numpy(proj_mat_numpy).to(self.device)



	
