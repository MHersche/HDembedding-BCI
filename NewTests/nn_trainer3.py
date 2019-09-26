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
from model1 import Net


class proj_trainer_end_end:
	
	def __init__(self,feat_dim,n_bands,HD_dim,n_classes,N_epoch,device,log_interval=50):
		
		self.feat_dim = feat_dim
		self.HD_dim = HD_dim
		self.device = device
		
		self._target_mem =  torch.ShortTensor(n_classes,self.HD_dim).bernoulli_().to(self.device)
	
		# Training settings
		parser = argparse.ArgumentParser(description='HD projection trainer')
		parser.add_argument('--batch-size', type=int, default=32, metavar='N',
							help='input batch size for training (default: 64)')
		parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
							help='input batch size for testing (default: 1000)')
		parser.add_argument('--epochs', type=int, default=N_epoch, metavar='N',
							help='number of epochs to train (default: 10)')
		parser.add_argument('--lr', type=float, default=64., metavar='LR', # 64.
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

		self.model = Net(feat_dim,n_bands,HD_dim,self.device).to(self.device)

		self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, momentum = self.args.momentum)
		self.criterion = F.binary_cross_entropy_with_logits


	def train(self,data,label):
		
		# convert target idx to real target 
		target = self._target_mem[label-1,:].float()
		data_cuda = torch.from_numpy(data).to(self.device).float()

		b_size = self.args.batch_size

		N_batches = int(np.floor(data.shape[0]/b_size))

		for epoch in range(self.args.epochs):
			for b_idx in range(N_batches):
				self.model.train()

				data_b = data_cuda[b_idx*b_size:(b_idx+1)*b_size]
				target_b = target[b_idx*b_size:(b_idx+1)*b_size]
				
				self.optimizer.zero_grad()
				output = self.model(data_b)
				loss = self.criterion(output, target_b) # negative log likelyhood loss 
				
				loss.backward()
				self.optimizer.step()
		
		
			# # testing 
			if (epoch % self.args.log_interval==0):
					self.test(data_cuda,target,epoch,True)

		return

	def test(self,data,target,epoch,do_print=False):
		self.model.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():

			output = self.model(data)
			test_loss = self.criterion(output, target) # negative log likelyhood loss 
			
			pred = (output > 0)
			accuracy = torch.mean(((pred.short()-target.short())==0).float())

		if do_print: 
			print('Epoch: {}, \t  Training set: Average loss: {:.4f}, Accuracy: {:.5f}'.format(epoch,
				test_loss, accuracy))
		return

	def get_params(self):
		proj_mat = self.model.get_weight()#self.model.fc1.weight.data#

		return self._target_mem, proj_mat,self.model.enc_vec.transpose(1,0) # self.model.enc_vec#
