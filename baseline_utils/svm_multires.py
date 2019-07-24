#!/usr/bin/env python3

''' 
LDA with variable precision 
'''

import numpy as np

from sklearn.svm import LinearSVC

__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"

class svm_multires(LinearSVC):

	def __init__(self,C = 1,intercept_scaling=1, loss='hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=1, tol=0.00001,precision=64):
		'''	LDA with self defined classification using quantized class vectors 
		Parameters
		----------
		sovler : string 
		shrinkage: string 
		precision: int
			{64,32,16}
		'''

		super().__init__(C = C,intercept_scaling=intercept_scaling, loss=loss, max_iter=max_iter,multi_class=multi_class, penalty=penalty, random_state=random_state, tol=tol)


		if precision==64: # use standard 
			self.score = self._quantscore
			self._dtype = np.float64
		elif precision == 32: 
			self.score = self._quantscore
			self._dtype = np.float32
		elif precision ==16: 
			self.score = self._quantscore
			self._dtype = np.float16
		else :
			raise ValueError('LDA invalid precision') 
			


	def _quantscore(self,X,y,sample_weight=None):


		y_hat = self._quantpredict(X)

		n_samples = y.shape

		score = np.sum(y_hat==y)/n_samples

		return score


	def _quantpredict(self,X):

		coef = self.coef_.astype(self._dtype)
		intercept = self.intercept_.astype(self._dtype)
		X = X.astype(self._dtype)
		est = np.matmul(X,np.transpose(coef,(1,0)))+intercept

		return np.argmax(est,axis=1)+1



		