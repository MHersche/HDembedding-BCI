#!/usr/bin/env python3

''' 
LDA with variable precision 
'''

import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"

class lda_multires(LinearDiscriminantAnalysis):

	def __init__(self,solver='svd',shrinkage=None,precision=64):		
		'''	LDA with self defined classification using quantized class vectors 
		Parameters
		----------
		sovler : string 
		shrinkage: string 
		precision: int
			{64,32,16}
		'''

		super().__init__(solver=solver,shrinkage=shrinkage)


		if precision==64: # use standard 
			self.score = self._quantscore
			self._dtype = np.float64
			self.predict= self._quantpredict
		elif precision == 32: 
			self.score = self._quantscore
			self.predict= self._quantpredict
			self._dtype = np.float32
		elif precision ==16: 
			self.score = self._quantscore
			self.predict= self._quantpredict
			self._dtype = np.float16
		elif precision ==2:
			self.score = self._biscore
			self.predict= self._bipredict
		else:
			raise ValueError('LDA invalid precision') 
	

	def _biscore(self,X,y,sample_weight = None):
		y_hat = self._bipredict(X)
		n_samples = y.shape
		score = np.sum(y_hat==y)/n_samples
		
		return score

	def _bipredict(self,X): 

		X = np.sign(X)
		coef = np.sign(self.coef_)
		#intercept = self.intercept_.astype(self._dtype)
		est = np.matmul(X,np.transpose(coef,(1,0)))

		return np.argmax(est,axis=1)+1	


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



		