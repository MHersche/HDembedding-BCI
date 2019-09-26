

import torch.nn as nn
import torch
import torch.nn.functional as F


class BinActive(torch.autograd.Function):
	'''
	Binarize the input activations and calculate the mean across channel dimension.
	'''
	def forward(self, input):
		self.save_for_backward(input)
		size = input.size()
		mean = torch.mean(input.abs(), 1, keepdim=True)
		input = (input.sign()+1)/2
		return input, mean

	def backward(self, grad_output, grad_output_mean):
		input, = self.saved_tensors
		grad_input = grad_output.clone()
		grad_input[input.ge(1)] = 0 # Gradient for input  >= 0 is 0 
		grad_input[input.le(-1)] = 0  # Gradient for input <= 0 is 0 
		return grad_input

class BipolarActive(torch.autograd.Function):
	'''
	Binarize the input activations and calculate the mean across channel dimension.
	'''
	def forward(self, input):
		self.save_for_backward(input)
		size = input.size()
		mean = torch.mean(input.abs(), 1, keepdim=True)
		input = input.sign()
		return input, mean

	def backward(self, grad_output, grad_output_mean):
		input, = self.saved_tensors
		grad_input = grad_output.clone()
		grad_input[input.ge(1)] = 0 # Gradient for input  >= 0 is 0 
		grad_input[input.le(-1)] = 0  # Gradient for input <= 0 is 0 
		return grad_input


class Net(nn.Module):
	'''
	spatial encoder w/ weight sharing 
	'''
	def __init__(self,feat_dim,n_bands,HD_dim,device):
		super(Net, self).__init__()
		
		self.n_bands = n_bands 
		self.HD_dim = HD_dim
		self.device = device
		
		self.enc_vec = torch.ShortTensor(n_bands,HD_dim).bernoulli_().to(self.device) # random 0,1 vector
		
		self.bp_enc_vec = self.enc_vec.float()*(-2)+1
		
		#my_eye = torch.eye(bp_enc_vec.size(1)).to(self.device)
		#c = bp_enc_vec.unsqueeze(2).expand(*bp_enc_vec.size(), bp_enc_vec.size(1))

		#self.enc_mat = torch.nn.Parameter( c*my_eye, requires_grad=False)
		# layers 
		self.fc1 = nn.Linear(feat_dim,HD_dim, bias = False)
		self.sigm= nn.Sigmoid()


	def forward(self, x):
		
		batch_size = x.shape[0]


		sum_vec = torch.Tensor(batch_size,self.HD_dim).zero_().to(self.device)


		for band_idx in range(self.n_bands): 

			x1 = self.fc1(x[:,band_idx])
			# simulate xor operation 
			x1 = x1 * self.bp_enc_vec[band_idx]

			x1,mean = BinActive()(x1)
			#x1 = self.sigm(x1)
			# randomly flip signs => 'xor'
			

			sum_vec.add_(x1) 

		#x1,mean = BinActive()(sum_vec-self.n_bands/2)
		x1 = sum_vec-int(self.n_bands/2)


		return x1

	def get_weight(self):
		
		return self.fc1.weight.data


class Net1(nn.Module):
	'''
	spatial encoder w/o weight sharing 
	'''


	def __init__(self,feat_dim,n_bands,HD_dim,device):
		super(Net1, self).__init__()
		
		self.n_bands = n_bands 
		self.HD_dim = HD_dim
		self.device = device
		self.feat_dim = feat_dim
		
		self.enc_vec = torch.ShortTensor(n_bands,HD_dim).bernoulli_().to(self.device) # random 0,1 vector
		
		self.bp_enc_vec = self.enc_vec.float()*(-2)+1
		
		#my_eye = torch.eye(bp_enc_vec.size(1)).to(self.device)
		#c = bp_enc_vec.unsqueeze(2).expand(*bp_enc_vec.size(), bp_enc_vec.size(1))

		#self.enc_mat = torch.nn.Parameter( c*my_eye, requires_grad=False)
		# layers 
		
		self.fc1 = nn.ModuleList([nn.Linear(feat_dim,HD_dim, bias = False) for i in range(n_bands)])

		self.sigm= nn.Sigmoid()


	def forward(self, x):
		
		batch_size = x.shape[0]


		sum_vec = torch.Tensor(batch_size,self.HD_dim).zero_().to(self.device)


		for band_idx in range(self.n_bands): 

			x1 = self.fc1[band_idx](x[:,band_idx])
			# simulate xor operation 
			x1 = x1 * self.bp_enc_vec[band_idx]

			x1,mean = BinActive()(x1)
			#x1 = self.sigm(x1)
			# randomly flip signs => 'xor'
			

			sum_vec.add_(x1) 

		#x1,mean = BinActive()(sum_vec-self.n_bands/2)
		x1 = sum_vec-int(self.n_bands/2)


		return x1


	def get_weight(self):

		out_tensor = torch.Tensor(self.n_bands,self.HD_dim,self.feat_dim).to(self.device)

		for band_idx in range(self.n_bands):
			out_tensor[band_idx]= self.fc1[band_idx].weight.data

		return out_tensor


class Net2(nn.Module):
	'''
	single encoder w/o weight sharing 
	'''

	def __init__(self,feat_dim,n_bands,HD_dim,device):
		super(Net2, self).__init__()
		
		if not((HD_dim % n_bands) ==0): 
			raise ValueError("HD dim = {:d} not multiple of n_bands = {:}".format(HD_dim,n_bands))  

		self.n_bands = n_bands 
		self.HD_dim = HD_dim
		self.device = device
		self.feat_dim = feat_dim
		self.subHD_dim = int(HD_dim/n_bands)
		
		self.enc_vec = torch.ShortTensor(HD_dim).zero_().to(self.device) # random 0,1 vector
		# layers 
		
		self.fc1 = nn.ModuleList([nn.Linear(feat_dim,self.subHD_dim, bias = False) for i in range(n_bands)])

		self.sigm= nn.Sigmoid()


	def forward(self, x):
		
		batch_size = x.shape[0]

		x1 = torch.Tensor(batch_size,self.HD_dim).zero_().to(self.device)

		for band_idx in range(self.n_bands): 

			x1[:,band_idx*self.subHD_dim:(band_idx+1)*self.subHD_dim] = self.fc1[band_idx](x[:,band_idx])

		return x1


	def get_weight(self):

		out_tensor = torch.Tensor(self.n_bands,self.subHD_dim,self.feat_dim).to(self.device)

		for band_idx in range(self.n_bands):
			out_tensor[band_idx]= self.fc1[band_idx].weight.data

		return out_tensor


class Net_knn(nn.Module):
	'''
	spatial encoder w/ weight sharing 
	'''
	def __init__(self,feat_dim,n_bands,HD_dim,device,n_inst):
		super(Net_knn, self).__init__()
		
		self.n_bands = n_bands 
		self.HD_dim = HD_dim
		self.device = device
		self.n_inst = n_inst
		
		self.enc_vec = torch.ShortTensor(n_bands,HD_dim).bernoulli_().to(self.device) # random 0,1 vector
		
		self.bp_enc_vec = self.enc_vec.float()*(-2)+1
		
		#my_eye = torch.eye(bp_enc_vec.size(1)).to(self.device)
		#c = bp_enc_vec.unsqueeze(2).expand(*bp_enc_vec.size(), bp_enc_vec.size(1))

		#self.enc_mat = torch.nn.Parameter( c*my_eye, requires_grad=False)
		# layers 
		self.fc1 = nn.Linear(feat_dim,HD_dim, bias = False)
		self.fc2 = nn.Linear(HD_dim,n_inst, bias = False)
		#self.fc2.requires_grad = False
		#self.fc2.weight.data = self.fc2.weight.data.sign()
		#self.fc2.requires_grad = False


	def forward(self, x):
		
		batch_size = x.shape[0]


		sum_vec = torch.Tensor(batch_size,self.HD_dim).zero_().to(self.device)


		for band_idx in range(self.n_bands): 

			x1 = self.fc1(x[:,band_idx])
			# simulate xor operation 
			x1 = x1 * self.bp_enc_vec[band_idx]

			x1,mean = BipolarActive()(x1)	

			sum_vec.add_(x1) 

		x1,mean = BipolarActive()(sum_vec)

		out = self.fc2(x1)/self.HD_dim

		return out,x1

	def get_weight(self):
		
		return self.fc1.weight.data

	def set_fc2_weight(self,weight,idx):

		#print(self.fc2.weight.data.shape,weight.shape)
		self.fc2.weight.data[idx] = weight	





