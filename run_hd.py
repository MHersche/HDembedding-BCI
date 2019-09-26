#!/usr/bin/env python3

''' 
Run HD classifier with different modifications and save them 
'''

import numpy as np
import time, sys 

import sys, os
from main_hd import Hd_model

DATA_PATH = './dataset/'
SAVE_PATH = './results/'
DATASET = "IV2a" # {IV2a,3classMI}
CROSSVAL = False
CLASSIFIER = 'assotiative'# {'assotiative', 'kmeans'}
# number of centroids if kmeans classifier 
k = [1]
# GPU device, if not supported takes automaically cpu 
CUDA_DEVICE = 'cuda:0' # 
# Feature 
feat_type = ['Riemann']# {"Riemann","Riemann_Euclid","Whitened_Euclid","CSP"}
# HD embedding 
code =['random_proj_gm']#{'learn_HD_proj_SGD','learn_HD_proj_ls',random_proj_gm','random_proj_bp','thermometer','greyish'}
# HD dimensions: depending on embedding different vector is taken 
HD_dim_vec1 = [100]#
HD_dim_vec2 = [10]#,
HD_dim_vec = {'learn_HD_proj_ls':HD_dim_vec1,'learn_HD_proj_SGD':HD_dim_vec1,'random_proj_bp':HD_dim_vec2,'random_proj_gm':HD_dim_vec2}
# number of bits per value in thermometer and grey coding => HD dimension = N_feat_per_band * d
d=[21] 
# sparsity in bipolar random projection 
sparsity = [0.9]#
encoding = ['spat'] #{'spat','single'}
learning = ['average'] # {'SVM','average'}

# LDA 
lda_svm_precision = 32


if DATASET == "IV2a":
	DATA_PATH = DATA_PATH + 'IV2a/'
	SAVE_PATH = SAVE_PATH + 'IV2a/'
	
	if feat_type[0] == 'CSP':
		N_feat_per_band = {'single': 216,'spat':24} 
	else: 
		N_feat_per_band = {'single':10879,'spat':253} 
	
	f_band = [np.arange(43)] # use all 43 frequency bands with BW 2,4,8,16,32 Hz
	time_windows = [np.array([[2.5,6]])] 

elif DATASET =='3classMI': 
	DATA_PATH = DATA_PATH + '3classMI/'
	SAVE_PATH = SAVE_PATH + '3classMI/'

	f_band = [np.arange(13)] # 2 Hz bands 4-30 Hz 
	time_windows = [np.array([[0,4]])]
	N_feat_per_band = {'single':1768,'spat':136}


############################################################

model = Hd_model(DATASET,DATA_PATH,CROSSVAL,CLASSIFIER,CUDA_DEVICE,lda_svm_precision=lda_svm_precision)

test_iter_vec = np.arange(1)

for test_iter in test_iter_vec:
	
	print('Test iteration number {}'.format(test_iter))
	result_nr = test_iter
	print(result_nr)

	for model.learning in learning:
		for model.encoding in encoding: 
			for model.feat_type in feat_type: 
				for model.code in code:
					for model.t_vec in time_windows:
						for model.f_band in f_band:
							for model.k in k:
								# matrix projections with dense entries 
								if model.code == 'random_proj_gm' or model.code == 'learn_HD_proj_ls' or model.code == 'learn_HD_proj_SGD' :
									
									model.N_feat_per_band = N_feat_per_band[model.encoding]
									for model.HD_dim in HD_dim_vec[model.code]: 
										model.save_path = SAVE_PATH + str(result_nr)
										model.run()
										result_nr += 1 
										
								# matrix projections with sparse entries 			
								elif model.code == 'random_proj_bp': 
									model.N_feat_per_band = N_feat_per_band[model.encoding]
									for model.HD_dim in HD_dim_vec[model.code]: 
										for model.sparsity in sparsity:
											model.save_path = SAVE_PATH + str(result_nr)
											model.run()
											result_nr += 1 

								# quantize based greyish, thermometer
								else:
									model.N_feat_per_band = N_feat_per_band[model.encoding]

									for model.d in d:		
										model.save_path = SAVE_PATH + str(result_nr)
										model.run()
										result_nr += 1 



