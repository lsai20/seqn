

import math
import os
import numpy as np


import scipy.io
import h5py
import pickle

import torch
import torch.nn as nn
#from torch.autograd import Variable
#from torch import optim
#import torch.nn.functional as F
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader, Dataset

import argparse
import torch

def parseargs():
	parser = argparse.ArgumentParser(description='Training deepsea model')

	# misc flags
	parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
	parser.add_argument('--verbose', action='store_true', help='Print additional status messages to console')

	# TODO specify logger or logging

	# specify data
	parser.add_argument('--data_mat_dir', type=str, default='deepsea_train/', help='Directory for DeepSea data in original .mat format')

	# training settings, defaults based on deepsea settings in run.sh or main.lua
	parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
	parser.add_argument('--epoch_size', type=int, default=110000, help='By default, deepsea cycles through subset of data. If negative, cycle through entire training data each epoch. TODO')
	parser.add_argument('--num_epochs', type=int, default=200, help='max num of epochs')

	# optimization settings, defaults based on deepsea
	parser.add_argument('--learning_rate', type=float, default=1, help='Initial learning rate')
	parser.add_argument('--learning_rate_decay', type=float, default = 8e-7)
	parser.add_argument('--weight_decay', type=float, default = 2 * 5e-7, help='Weight decay for SGD, equal to 2 * L2')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
	parser.add_argument('--stdv', type=float, default=0.05, help='Standard deviation for initializing parameters')
	parser.add_argument('--L1_sparsity', type=float, default=1e-08, help='L1 penalty to last hidden layer')
	parser.add_argument('--max_kernel_norm', type=float, default=0.9, help='Constrain kernel norm to this value. If negative value, do not constrain')

	args = parser.parse_args()
	return args

# TODO do i need to pass these parts of model to cuda? e.g. nn.conv1d(...).cuda() 


class SeqDataset(Dataset):
	def __init__(self, matFile, args=None):

		self.args = args
		self.config = config

		# load data from .mat (train, test, or valid)
		# self.labels1hot is 1 hot of 919 features for each sample
		# self.inputs1hot is 1 hot of ACGT for each sample, dim (num seq, seq len, 4)
		
		datasetName = matFile.rsplit('/')[-1].split('.')[0]
		try: # test.mat and valid.mat are MATLAB 5.0 file
			print('loading %s as matlab file' % datafile)
			datamat = scipy.io.loadmat(datafile)
			self.labels1hot = datamat[datasetName + 'data']
			self.inputs1hot = datamat[datasetName + 'xdata']
		except NotImplementedError: # train.mat is MATLAB 7.3 file
			# use HDF reader for matlab v7.3 files
			# get array from HDF5 dataset  with [::]
			print('try opening data as h5df')
			f = h5py.File(datafile, 'r')
			self.labels1hot = f[datasetName + 'data'][::]
			self.inputs1hot = f[datasetName + 'xdata'][::]
			f.close()		

		
		self.seqlen = self.inputs1hot.shape[1] # length of seq window
		# could add option to trim input to desired size

	def __len__(self): # number of samples in dataset
		return len(self.labels1hot)

	def __getitem__(self, item):
		# note: both input and label of deepsea are int
		# some later functions may require LongTensor
		return (torch.tensor(self.labels1hot[item]),
						torch.tensor(self.inputs1hot[item])
					 )



class DeepSeaModel (nn.Module): 
	''' deepsea CNN based on lua source code, reimplemented in pytorch
	'''

	def __init__(self): 
		super ( DeepSeaModel, self).__init__()
		self.threshold = 1e-06 # DeepSea uses RELU, but in their lua source code, the threshold is 1e-6 instead of 0

		#self.num_of_feature = 1000
		#self.input_channels = 4
		# number of channels after the three convolution layers (each with window size 8, no padding) and two maxpools with stride 4
		nchannel_fc = math.floor((math.floor((1000-7)/4.0)-7)/4.0)-7

		# input is 4 features (one hot) x 1000 bp
	 	# layer 1: convolution layer with 4 features, 320 kernels, window size 8, step size 1
	 	# layer 2: pooling with window size 4, step size 4; drop out p 0.2
		# layer 3: convolution with 480 kernels, window size 8, step size 1
		# layer 4: pooling, window size 4, step size 4, drop out p 0.2
		# layer 5: convolution with 960 kernels, window size 8, step size 1; p dropout 0.5
		# layer 6:	reshape, fully connected layer with 925 neurons
		# layer 7: sigmoid, final output is 919 features

		self.all_layers = nn.Sequential ( 
			nn.Conv1d( 4, 320, 8, stride=1),
			nn.Threshold(0, self.threshold),

			nn.MaxPool1d(4, 4), 
			nn.Dropout(p=0.2),

			nn.Conv1d(320, 480, 8, stride=1),
			nn.Threshold(0, self.threshold),

			nn.MaxPool1d(4, 4),
			nn.Dropout(p=0.2),

			nn.Conv1d(480, 960, 8, stride=1),	
			nn.Threshold(0, self.threshold),
			nn.Dropout(p = 0.5),

			nn.Flatten(),
			nn.Linear(nchannel_fc*960, 925),
			nn.Threshold(0, self.threshold),

			nn.Linear(925, 919),
			nn.Sigmoid()
			)

	def forward(self, x):
		# could also return a tuple,l like hugging face
		return self.deepsea_cnn(x)
			# should return prediction
			# take derivative of loss in backward, binary loss

	#def loss()
	# todo implement loss function




# optimization settings based on DeepSea run.sh in source code
def train(model, train_dataset, valid_dataset, args):
	'''pass in (potentially pretrained) model on correct device, Dataset objects, and dict of args'''
	'''
	learning_rate = 1
	learning_rate_decay = 8e-7
	weight_decay = 2 * 5e-7	# weight decay is 2 * L2
	momentum = 0.9
	stdv = 0.05
	L1_sparsity = 1e-08
	max_kernel_norm = 0.9
	'''

	# DeepSea uses SGD with momentum
	# TODO max kernel norm and l1 penaltyyyy
	optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay)
	# TODO do i need a scheduler? 
	

	# Note: deepsea shuffles data every epoch. Currently doing plain training
	num_batches = int(len(train_dataset) / float(batch_size)) + 1
	train_loader = torch.utils.data.DataLoader(train_dataset,
		batch_size = args.batch_size, shuffle = True, num_workers = 4)

	model.train(); model.zero_grad() # set to training mode, clear gradient
	tr_loss = 0.0	# training loss

	train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

	for epoch in train_iterator:

		epoch_iterator = tqdm(train_dataloader, desc="Iteration")

		current_loss = 0
		num_batches_done = 0

		for step, batch in enumerate(epoch_iterator):
			inputs = inputs.to(args.device)
			labels = labels.to(args.device)
			outputs = model(inputs) # forward
			loss = nn.CrossEntropyLoss(outputs, labels) # backward 
			loss.backward()
			optimizer.step() # optimization
			model.zero_grad()

			tr_loss += loss.item()

			num_batches_done += 1
			print('epoch, batch', epoch, num_batches_done)
			print('outputs', outputs)
			print('tr_loss', tr_loss)
		
		# end-of-epoch report
		# training loss, LATER: train_acc, valid_loss, valid_acc


	return global_step, tr_loss



def evaluate(model, eval_data, args):
	'''evaluate model, e.g. on validation or test data'''
	return


def main():
	args = parseargs()	# handle user arguments
	print(args)

	args.device = None
	if not args.disable_cuda and torch.cuda.is_available():
			args.device = torch.device('cuda') # specify 'cuda:0' ?
	else:
			args.device = torch.device('cpu')
	print('Using device: %s' % args.device)
	
	
	##### DATA #####
	# comments: 919 marks include 690 TF binding profiles (for 160 TFs), 125 DHS profiles, 104 histone-mark profiles
	# 17% of genome bound by at least one measured TF, unclear if they only used that 17% for training
	train_dataset = SeqDataset(os.path.join(args.data_mat_dir, 'train.mat'))
	valid_dataset = SeqDataset(os.path.join(args.data_mat_dir, 'valid.mat'))


	##### INITIALIZE MODEL #####
	deepsea_model = DeepSeaModel()
	#if not args.disable_cuda:
	#	deepsea_model.cuda() # move to gpu before making optimizer


	##### TRAIN #####
	train(model, train_dataset, valid_dataset, args)
	



	# input a batch
	x = torch.FloatTensor(test_mat['testxdata'][0]).unsqueeze(0)
	print('x shape', x.shape, x[0:3])
	# do a forward pass on one input (batch size 1)
	deepsea_model = DeepSeaModel()
	print(deepsea_model.all_layers[0].weight.grad)
	y = deepsea_model.forward(x)
	todo_loss.backward() # should be loss backward
	print(deepsea_model.deepsea_cnn[0].weight.grad)
	print(y.shape)
	print(y)


if __name__ == '__main__':
	main()
#
