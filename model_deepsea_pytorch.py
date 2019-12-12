

import math
import os
import numpy as np


import scipy.io
import h5py
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader, Dataset

import argparse

from tqdm import tqdm, trange
from scipy.sparse import coo_matrix # not useful for 3d array

import sklearn.metrics

import logging
logger = logging.getLogger(__name__)


# TODO add logging
# TODO check code matches the lua, e.g. L1, num epochs

# comments on Deepsea data:
#  919 marks include 690 TF binding profiles (for 160 TFs), 125 DHS profiles, 104 histone-mark profiles
# 17% of genome bound by at least one measured TF, unclear if they only used that 17% for training


def parseargs():
	parser = argparse.ArgumentParser(description='Training deepsea model')

	# misc flags
	parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
	parser.add_argument('--verbose', action='store_true', help='Print additional status messages to console')
	parser.add_argument('--mini_data_set', action='store_true', help='load test data instead of full train set for debugging TODO')
	# specify data
	parser.add_argument('--data_mat_dir', type=str, default='deepsea_train/', help='Directory for DeepSea data in original .mat format')
	parser.add_argument('--output_dir', type=str, default='deepsea_output/', help='Directory containing sparse matrices of data')

	# option to load pretrained statedict 
	# could also add option to save checkpoint instead of savedict
	parser.add_argument('--load_statedict_name', type=str, default = None, help = 'Pretrained statedict to load.')


	# training settings, defaults based on deepsea settings in run.sh or main.lua
	parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
	parser.add_argument('--epoch_size', type=int, default=110000, help='By default, deepsea cycles through subset of data. If negative, cycle through entire training data each epoch. TODO')
	parser.add_argument('--num_train_epochs', type=int, default=300, help='max num of epochs if loss not decreasing')

	# evaluation settings
	parser.add_argument('--eval_batch_size', type=int, default=16, help='test/validation batch size')

	# optimization settings, defaults based on deepsea
	parser.add_argument('--learning_rate', type=float, default=1, help='Initial learning rate')
	parser.add_argument('--adam_learning_rate', type=float, default=0.0001, help='Initial learning rate')

	parser.add_argument('--learning_rate_decay', type=float, default = 8e-7)
	parser.add_argument('--weight_decay', type=float, default = 2 * 5e-7, help='Weight decay for SGD, equal to 2 * L2')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
	parser.add_argument('--stdv', type=float, default=0.05, help='Standard deviation for initializing parameters')
	parser.add_argument('--L1_sparsity', type=float, default=1e-08, help='L1 penalty to last hidden layer')
	parser.add_argument('--max_kernel_norm', type=float, default=0.9, help='Constrain kernel norm to this value. If negative value, do not constrain')

	args = parser.parse_args()
	return args



class SeqDataset(Dataset):
	def __init__(self, datasetName, matFile=None, args=None):

		self.args = args

		# load data from .mat (train, test, or valid)
		# self.labels1hot is 1 hot of 919 features for each sample, e.g. (N, 919)
		# self.inputs1hot is 1 hot of ACGT for each sample, dim (num seq, 4, seq len), e.g. (N, 1000, 4)
		
		if not matFile:
			print('Initializing empty SeqDataset obj')
			self.labels1hot = None
			self.inputs1hot = None
			self.seqlen = None

		else:
			# datasetName is train/test/valid
			datasetName = matFile.rsplit('/')[-1].split('.')[0]
			try: # test.mat and valid.mat are MATLAB 5.0 file
				print('loading %s as matlab file' % matFile)
				datamat = scipy.io.loadmat(matFile)
				self.labels1hot = datamat[datasetName + 'data']
				self.inputs1hot = datamat[datasetName + 'xdata']
			except NotImplementedError: 
				# train.mat is MATLAB 7.3 file so use HDF reader
				# get array from HDF5 dataset	with [::]
				# note: dim of train.mat is (seq len, 4, num seq) and (919, num seq), so need to transpose
				print('try opening data as h5df')
				f = h5py.File(matFile, 'r')
				self.labels1hot = f[datasetName + 'data'][::].transpose()
				self.inputs1hot = f[datasetName + 'xdata'][::].transpose((2,1,0))
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
			nn.Dropout(p = 0.2),

			nn.Conv1d(320, 480, 8, stride=1),
			nn.Threshold(0, self.threshold),

			nn.MaxPool1d(4, 4),
			nn.Dropout(p = 0.2),

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
		return self.all_layers(x)
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
	# TODO max kernel norm and l1 penalty
	#optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay)
	optimizer = optim.Adam(model.parameters(), lr=args.adam_learning_rate)


	# Note: deepsea shuffles data every epoch. Currently doing plain training
	num_batches = int(len(train_dataset) / float(args.batch_size)) + 1


	train_dataloader = torch.utils.data.DataLoader(train_dataset,
		batch_size = args.batch_size, 
		shuffle = True, 
		num_workers = 4)

	loss_fxn = nn.BCELoss() # . deepsea train uses BCE for multilabel

	model.train(); model.zero_grad() # set to training mode, clear gradient
	tr_loss = 0.0	# training loss

	logger.info("***** Running training *****")
	logger.info("	Num examples = %d", len(train_dataset))
	logger.info("	Num Epochs = %d", args.num_train_epochs)
	
	# save args, set up dir for saving statedicts (or checkpoints)
	torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
	if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)


	## track best loss on valid set 
	best_eval_loss = np.inf
	last_best = 0
	break_early = False

	train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

	global_step = 0.0 # how many batches total across epochs

	for epoch in train_iterator:

		epoch_iterator = tqdm(train_dataloader, desc="Epoch%dTrainIteration" % epoch)

		current_loss = 0
		num_batches_done = 0
		tr_loss = 0.0	# training loss
		logging_loss = 0.0 # ???


		# note: the 1 hots inputs are ByteTensors, either make SeqDataset as float or convert batch by batch
		# BCE loss expects labels to be float, CE expects Long
		for step, batch in enumerate(epoch_iterator):
			#if step > args.epoch_size: # cycle through subset of data each epoch (messes up the iterator display to break early)
			#print('reached epoch size, now evaluating')
			#break

			labels = batch[0].float().to(args.device)
			inputs = batch[1].float().to(args.device)

			outputs = model(inputs) # forward
			loss = loss_fxn(outputs, labels) # backward 
			loss.backward()
			optimizer.step() # update based on grad
			#model.zero_grad()
			optimizer.zero_grad()
			tr_loss += loss.item()

			num_batches_done += 1
			global_step += 1 # not sure if this is right

		# end-of-epoch report
		# currently print f1 and roc_auc in evaluate fxn
		
		print('\n#######################\nepoch %d, tr_loss/batch_size %f' % (epoch, tr_loss/args.batch_size))

		# eval on training set is slow, either eval on sbuset or skit
		#print('\nTraining evaluate')
		#result_tr = evaluate(model, train_dataset, args, prefix="%d_%d" % (epoch, global_step), verbose=True)
		#tr_loss2, tr_f1_by_label, tr_roc_auc_by_label, tr_avg_pr_by_label = result_tr

		print('\nValidation evaluate')
		result_eval = evaluate(model, valid_dataset, args, prefix="%d_%d" % (epoch, global_step), verbose=True, iterator_desc="Epoch_%d_evaluating" % epoch)
		eval_loss, eval_f1_by_label, eval_roc_auc_by_label, eval_avg_pr_by_label = result_eval
		
		
		print('\n\ntrain, valid loss %f, %f' % (tr_loss/global_step, eval_loss) )
		# save statedict
		#model_to_save.save_pretrained(output_dir) #??? huggingface?
		logger.info("Saving model statedict (not checkpoint?) to %s", args.output_dir)
		torch.save(model.state_dict(), os.path.join(args.output_dir, 'training_statedict_epoch_%d.dict' % (epoch) ))
	


		if eval_loss < best_eval_loss:
			best_eval_loss = eval_loss
			last_best = epoch
			break_early = False
			print ('\nupdate lowest loss: epoch {}, loss {}\nreset break_early to False, see break_early variable {}'.format(epoch,best_eval_loss,break_early))
			torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_training_statedict_epoch_%d.dict' % (epoch) ))

			logger.info("Saving model statedict (not checkpoint?) to %s", args.output_dir)

		else:
			if epoch - last_best > 5 : ## break counter after some epoch, changed from 3 to 5
				break_early = True
				print ('epoch {} set break_early to True, see break_early variable {}'.format(epoch,break_early))

		if break_early:
			train_iterator.close()
			print ("**** break early ****")
			break
		

		model.train(); # set back to training mode
		print('Epoch %d end\n###################################\n\n\n', flush=True)

	return tr_loss


# TODO use the evaluation prefix or get rid of arg
def evaluate(model, eval_dataset, args, prefix="", verbose=False, iterator_desc="Evaluating"):
	'''evaluate model, e.g. on validation or test data'''

	eval_output_dir = args.output_dir
	# TODO use the outputdir
	if not os.path.exists(eval_output_dir):
		#os.makedirs(eval_output_dir)
		print('Specified output dir doesn\'t exist: %s' % args.output_dir)
		exit(1)
	
	# Note that DistributedSampler samples randomly
	#eval_sampler = SequentialSampler(eval_dataset) # may not even need sampler

	eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size)

	logger.info("***** Running evaluation {} *****".format(prefix))
	logger.info("	Num examples = %d", len(eval_dataset))
	logger.info("	Batch size = %d", args.eval_batch_size)
	eval_loss = 0.0
	model.eval()


	loss_fxn = nn.BCELoss() # . deepsea train uses BCE for multilabel, in 3_loss.lua. could add arg for loss function, or make loss part of the model class

	nb_eval_steps = 0.0
	num_samples_so_far = 0
	# TODO step, batch?
	all_outputs = np.zeros_like(eval_dataset.labels1hot, dtype=float)

	for batch in tqdm(eval_dataloader, desc=iterator_desc):
		# convert byte to float
		labels = batch[0].float().to(args.device)
		inputs = batch[1].float().to(args.device)

		with torch.no_grad():
			outputs = model(inputs)
			eval_loss += loss_fxn(outputs, labels).item()

		
		# can use dat/huggingface evaluation metric to get pr@k

		# append outputs for this batch
		i1 =  num_samples_so_far
		i2 = num_samples_so_far + len(labels)
		all_outputs[i1:i2,:] = outputs.cpu() # all_outputs is np arr
		num_samples_so_far += len(labels)

		nb_eval_steps += 1

	eval_loss = eval_loss / (nb_eval_steps*args.eval_batch_size) # normalize by num samples
	all_labels = eval_dataset.labels1hot
	# note that sklearn requires numpy, not tensor

	# compute f1 and roc_auc for each label
	# (could compute all peak types at once and average, but sklearn.metrics throws error for labels with only one class)
	eval_f1_by_label = np.zeros(919)
	eval_roc_auc_by_label = np.zeros(919)
	eval_avg_pr_by_label = np.zeros(919)

	# round to get predictions at 0.5 thresh
	all_outputs_rounded = np.rint(all_outputs).astype(int)

	'''
	if verbose:
		print('\nnum pos per class, true')
		print(sum(all_labels))
		print('num pos per class, predicted')
		print(sum(all_outputs))
		print('num pos per class, predicted, rounded')
		print(sum(all_outputs_rounded))
		print('\n')
	'''
	if verbose:
		print('\nmarker\tf1_score\troc_auc\tavg_pre')
	
	for marker_idx in range(919):
		# check for case with only one class
		num1s = sum(all_labels[:,marker_idx])
		if  num1s == 0 or num1s == 919: 
			if verbose:
				print('No pos label, marker %d' % marker_idx ) 
			eval_f1_by_label[marker_idx] = np.nan
			eval_roc_auc_by_label[marker_idx] = np.nan
			eval_avg_pr_by_label[marker_idx] = np.nan

		else:
		# if no positive predictions, f1 set to 0 (sklearn 0.22 has option zero_division=0 for warnings surpressed). do it manual here for 0.21
			if sum(all_outputs_rounded[:,marker_idx]) == 0:
				eval_f1_by_label[marker_idx] = 0
			else:
				eval_f1_by_label[marker_idx] = sklearn.metrics.f1_score(all_labels[:,marker_idx], all_outputs_rounded[:,marker_idx], average='binary')

			eval_roc_auc_by_label[marker_idx] = sklearn.metrics.roc_auc_score(all_labels[:,marker_idx], all_outputs[:,marker_idx])
			eval_avg_pr_by_label[marker_idx] = sklearn.metrics.average_precision_score(all_labels[:,marker_idx], all_outputs[:,marker_idx])

			if verbose:
				print('%d:\t%f\t%f\t%f' % (marker_idx, eval_f1_by_label[marker_idx], eval_roc_auc_by_label[marker_idx], eval_avg_pr_by_label[marker_idx]))
	
	# TODO add option to save outputs and metrics by peak type
	if verbose:
		print('\nLoss %f' % eval_loss)
		#print('f1')
		#print(eval_f1_by_label[0:5])
		#print('auc')
		#print(eval_roc_auc_by_label[0:5])

	#print('(micro) mean eval_f1')
	#print(np.mean(eval_f1_by_label)) 
	#print('(micro) mean eval_roc_auc')
	#print(np.mean(eval_roc_auc_by_label))

	# TODO also return avg_precision and give option to return or save predictions


	return eval_loss, eval_f1_by_label, eval_roc_auc_by_label, eval_avg_pr_by_label




def main():
	args = parseargs()	# handle user arguments
	print(args)

	args.device = None
	if not args.disable_cuda and torch.cuda.is_available():
			args.device = torch.device('cuda') # specify 'cuda:0' ?
	else:
			args.device = torch.device('cpu')
	print('Using device: %s' % args.device)
	
	




	#### LOAD PRETRAINED MODEL AND EVALUATE ####
		# load saved model and test data only
	# could also check automatically if model ends with .pt or .pth

	test_dataset = SeqDataset('test', matFile = os.path.join(args.data_mat_dir, 'test.mat'))
	print('test_dataset input and label shapes')
	print(test_dataset.inputs1hot.shape)
	print(test_dataset.labels1hot.shape)


	if args.load_statedict_name:
		# get architecture, reset final layer to one output for each class
		deepsea_model = DeepSeaModel()
		if not args.disable_cuda:
			deepsea_model.cuda()

		deepsea_model.load_state_dict(torch.load(args.load_statedict_name))

		if args.verbose:
			print('Loaded saved model from state dict')


	# Else load train data and train
	elif args.load_statedict_name == None or args.load_statedict_name == '':
		##### TRAIN/VALID DATA AND TEST #####

		if args.mini_data_set:
			train_dataset = SeqDataset('test',  matFile = os.path.join(args.data_mat_dir, 'test.mat'))
			# TODO add constructor for SeqDataset to init from array
			train_dataset.labels1hot = train_dataset.labels1hot[2000:3000]
			train_dataset.inputs1hot = train_dataset.inputs1hot[2000:3000]

		else:
			train_dataset = SeqDataset('train', matFile = os.path.join(args.data_mat_dir, 'train.mat'))

		print('\ntrain_dataset input and label shapes')
		print(train_dataset.inputs1hot.shape)
		print(train_dataset.labels1hot.shape)
		
		valid_dataset = SeqDataset('valid',  matFile = os.path.join(args.data_mat_dir, 'valid.mat'))
		print('\nvalid_dataset input and label shapes')
		if args.mini_data_set:
			valid_dataset.labels1hot = valid_dataset.labels1hot[0:1000]
			valid_dataset.inputs1hot = valid_dataset.inputs1hot[0:1000]

		print(valid_dataset.inputs1hot.shape)
		print(valid_dataset.labels1hot.shape)

		test_dataset = SeqDataset('test',  matFile = os.path.join(args.data_mat_dir, 'test.mat'))
		print('\ntest_dataset input and label shapes')
		print(test_dataset.inputs1hot.shape)
		print(test_dataset.labels1hot.shape)

		print('\n\n\n\n\n', flush=True)


		##### INITIALIZE MODEL AND TRAIN #####
		# currently save best-so-far model during train(...)
		deepsea_model = DeepSeaModel()
		if not args.disable_cuda:
			deepsea_model.cuda() # move to gpu before making optimizer

		print('\n############# TRAINING ########################')
		train(deepsea_model, train_dataset, valid_dataset, args)
			


	#### RUN ON TEST DATA ####
	deepsea_model.eval() # set to evaluation mode
	# TODO, temporarily set test data to be small, roc_roc and avg_pr very slow on server
	test_dataset.labels1hot = test_dataset.labels1hot[0:10000]
	test_dataset.inputs1hot = test_dataset.inputs1hot[0:10000]
	print('\n############### EVAL ON 10k TEST SAMPLES #############', flush=True)
	evaluate(deepsea_model, test_dataset, args, prefix="", verbose=True, iterator_desc="Evaluating Test")



'''
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
'''



if __name__ == '__main__':
	main()
#
