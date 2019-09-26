#!/usr/bin/env python3

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.svm import LinearSVC,SVC

import sys, os
from model1 import Net_knn


class proj_trainer_unsupervised:
	
	def __init__(self,feat_dim,n_bands,HD_dim,n_classes,N_epoch,device,n_inst,log_interval=10):
		
		self.feat_dim = feat_dim
		self.HD_dim = HD_dim
		self.device = device
		self.n_classes=n_classes
		
		self.k = 5
		
		self._target_mem =  torch.ShortTensor(n_classes,self.HD_dim).bernoulli_().to(self.device)
	
		# Training settings
		parser = argparse.ArgumentParser(description='HD projection trainer')
		parser.add_argument('--batch-size', type=int, default=32, metavar='N',
							help='input batch size for training (default: 64)')
		parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
							help='input batch size for testing (default: 1000)')
		parser.add_argument('--epochs', type=int, default=N_epoch, metavar='N',
							help='number of epochs to train (default: 10)')
		parser.add_argument('--lr', type=float, default=1, metavar='LR', # 64.
							help='learning rate (default: 0.01)')
		parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
							help='SGD momentum (default: 0.5)')
		parser.add_argument('--no-cuda', action='store_true', default=False,
							help='disables CUDA training')
		parser.add_argument('--seed', type=int, default=1, metavar='S',
							help='random seed (default: 1)')
		parser.add_argument('--log-interval', type=int, default=log_interval, metavar='N',
							help='how many batches to wait before logging training status')
		self.args = parser.parse_args()
		
		torch.manual_seed(self.args.seed)

		self.model = Net_knn(feat_dim,n_bands,HD_dim,self.device,n_inst).to(self.device)
		
		self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, momentum = self.args.momentum)
		self.criterion = F.binary_cross_entropy_with_logits


	def train(self,data,label):
		
		# convert target idx to real target 
		target = F.one_hot(torch.arange(data.shape[0])).float()
		label_cuda = torch.from_numpy(label).to(self.device)

		data_cuda = torch.from_numpy(data).to(self.device).float()

		b_size =data.shape[0]# self.args.batch_size

		N_batches = int(np.floor(data.shape[0]/b_size))

		for epoch in range(self.args.epochs):
			# # testing 
			if (epoch % self.args.log_interval==0):
				self.test(data_cuda,target,label_cuda,epoch,True)
			for b_idx in range(N_batches):
				self.model.train()
				idx = torch.arange(b_idx*b_size,(b_idx+1)*b_size)
				data_b = data_cuda[idx]
				target_b = target[idx]
				
				self.optimizer.zero_grad()
				output,vec = self.model(data_b)
				
				#print(target_b.shape,data_b.shape)
				
				loss = self.criterion(output, target_b) # negative log likelyhood loss 
				
				loss.backward()
				self.optimizer.step()

				#self.model.set_fc2_weight(vec,idx)

		return

	def test(self,data,target,target_label,epoch,do_print=False):
		self.model.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():

			output,vec = self.model(data)
			test_loss = self.criterion(output, target) # negative log likelyhood loss 
			
			pred = self.knn_predict(output,target_label)
			accuracy = torch.mean(((pred.short()-target_label.short())==0).float())

			if do_print: 
				print('Epoch: {:}, \t  Training set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch,
					test_loss, accuracy))
		return


	def knn_predict(self,X,y):
		'''	Predict multiple samples

		'''	

		
		NO_samples = X.shape[0] 	
		y_hat = torch.Tensor(NO_samples)
		for smpl_idx in range(0,NO_samples):
			
			cos_sim = X[smpl_idx]/self.HD_dim
			y_hat[smpl_idx] = self.knn_class(cos_sim,y)

		return y_hat

	def knn_class(self,sim,y):
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
        	        
        	max_idx = torch.argsort(sim)[-self.k:]
        
        	max_sim = sim[max_idx]
        	max_label = y[max_idx]
        	score = torch.Tensor(self.n_classes).to(self.device).zero_() 
        	for clas in range(self.n_classes): 
            		clas_idx = (max_label == (clas+1))
            		score[clas] = torch.sum(torch.exp(max_sim[clas_idx]))
            
        	score = score/torch.sum(score)
        	y_hat = torch.argmax(score)+1
        	return y_hat 


	def get_params(self):
		proj_mat = self.model.get_weight()#self.model.fc1.weight.data#

		return self._target_mem, proj_mat,self.model.enc_vec.transpose(1,0) # self.model.enc_vec#
