#!/usr/bin/env python3

''' 
import features either CSP or Riemannian 
'''

import numpy as np

import sys, os

import time

from get_data_epfl import get_data_epfl
from filters import load_filterbank 
from riemannian_multiscale import riemannian_multiscale


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"


def load_feature_EPFL(data_path, subject, twind_sel,band_sel, feat_type, encoding, split=0,n_splits = 0,save_features=False,load_features=False):
	'''	returns features and labels of training and test set 

	Keyword arguments:
	subject -- Number of subject in [1,2,...,5]
	band_sel-- selection of frequency bands used 
	twind_sel -- selection of timewindows use , depends on CSP or R 
	feat_type -- {"Riemann","Whitened_Euclid"}
	encoding -- 'single', 'spat', 'temp_spat'
	

	Return:	train_feat -- numpy array with dimension [trial, temp_wind, bands, R dim ]
			train_label -- numpy array with entries [1,2,3]

			test_feat -- numpy array with dimension [trial, temp_wind, bands, R dim ]
			test_label -- numpy array with entries [1,2,3]

	'''
	NO_channels = 16 # number of EEG channels 
	NO_subjects = 5
	NO_riem = int(NO_channels*(NO_channels+1)/2) # Total number of Riemannian feature per band and timewindow

	sub_path =  '/S'+str(subject)+'/' +feat_type  + str(split) 
	path = data_path + sub_path 
	
	if load_features: 
		#print("Load precalculated features")
		with np.load(path+ '.npz') as data:
			train_feat = data['train_feat']
			train_label = data['train_label'].astype(int) 
			test_feat = data['test_feat']
			test_label = data['test_label'].astype(int)

		N_tr_trial = train_feat.shape[0] 
		N_test_trial = test_feat.shape[0]
		N_bands = len(band_sel)
		N_twind = len(twind_sel)

		# select necessairy bands and twindows
		#train_feat = train_feat[np.ix_(np.arange(N_tr_trial),twind_sel,band_sel)].reshape(N_tr_trial,N_twind,N_bands,NO_riem)
		#test_feat = test_feat[np.ix_(np.arange(N_test_trial),twind_sel,band_sel)].reshape(N_test_trial,N_twind,N_bands,NO_riem)

	else: 
		print('Generate new set of features')
		train_feat,train_label,test_feat,test_label = generate_Riemann_feat(data_path,subject,twind_sel ,band_sel,feat_type,split)	
		twind_sel = np.arange(twind_sel.shape[0])

	# get number of trials 
	N_tr_trial = train_feat.shape[0] 
	N_test_trial = test_feat.shape[0]
	N_bands = len(band_sel)
	N_twind = len(twind_sel)


	if save_features:
		np.savez(path,train_feat = train_feat,train_label = train_label,test_feat = test_feat, test_label=test_label)

	if encoding == 'single':
		train_feat = np.reshape(train_feat,(N_tr_trial,N_twind,-1) )
		svm_train_feat = np.reshape(train_feat,(N_tr_trial,-1))
		test_feat = np.reshape(test_feat,(N_test_trial,N_twind,-1))
		svm_test_feat = np.reshape(test_feat,(N_test_trial,-1))

	elif encoding == 'spat' or encoding =='spat_bind':
		train_feat = np.reshape(np.transpose(train_feat,(0,2,1,3)),(N_tr_trial,1,N_bands,-1))
		test_feat = np.reshape(np.transpose(test_feat,(0,2,1,3)),(N_test_trial,1,N_bands,-1))
		# reshape features for svm 
		svm_train_feat = np.reshape(train_feat,(N_tr_trial,-1))
		svm_test_feat = np.reshape(test_feat,(N_test_trial,-1))

	


	return train_feat,svm_train_feat, train_label, test_feat,svm_test_feat, test_label



def generate_Riemann_feat(data_path,subject,twind_sel,band_sel,riem_settings,fold):

	################################### SETTINGS ######################################################
	fs = 512. # sampling frequency 
	NO_channels = 16 # number of EEG channels 
	NO_riem = int(NO_channels*(NO_channels+1)/2) # Total number of CSP feature per band and timewindow
	bw = np.array([2,4,8,16,32]) # bandwidth of filtered signals 
	ftype = 'butter' # 'fir', 'butter'
	forder= 1 # 4
	filter_bank = load_filterbank(bw,fs,order=forder,max_freq=30,ftype = ftype) # get filterbank coeffs 
	
	time_windows = (twind_sel*fs).astype(int)

	# restrict time windows and frequency bands 
	# self.time_windows = self.time_windows[10] # use only largest timewindow
	
	filter_bank = filter_bank[band_sel] # use only 2Hz bands 4-30 Hz
	rho = 0.1

	N_bands = filter_bank.shape[0]
	N_time_windows = time_windows.shape[0]
	NO_features = NO_riem*N_bands*N_time_windows
	
	#####################################################################################################

	# load time sample data 
	train_data,train_label,test_data,test_label = get_data_epfl(data_path,subject,fold)

	# 1. calculate features and mean covariance for training 
	riemann = riemannian_multiscale(filter_bank,time_windows,riem_opt =riem_settings,rho = rho,vectorized = False)
	train_feat = riemann.fit(train_data)

	# 2. Testing features 
	test_feat = riemann.features(test_data)

	# Rescale of features (based on training data)
	mean = np.mean(train_feat,axis=0).reshape(1,N_time_windows,N_bands,-1)
	sigma = np.std(train_feat,axis = 0).reshape(1,N_time_windows,N_bands,-1)
	train_feat = (train_feat - mean)/sigma
	test_feat = (test_feat - mean)/sigma

	return train_feat,train_label,test_feat,test_label