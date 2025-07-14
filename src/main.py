# ===== monkey‑patch for Python 3.10+ compatibility =====
import collections
import collections.abc
collections.Mapping        = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence       = collections.abc.Sequence
# ======================================================

import os
import sys
import math
import logging
import ipdb as pdb
import random
import numpy as np
from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
import copy
try:
	import cPickle as pickle
except ImportError:
	import pickle

from src.args import build_parser
from src.utils.helper import *
from src.utils.logger import get_logger, print_log, store_results
from src.dataloader import DyckCorpus, Sampler, CounterCorpus
from src.model import LanguageModel, build_model, train_model, run_validation, run_test
from src.utils.dyck_generator import DyckLanguage


global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = 'logs'
model_folder = 'models'
result_folder = './out/'
data_path = './data/'
board_path = './runs/'

def load_data(config, logger, voc = None):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging
			voc (Vocabulary Object) : Provided only during test time

		Returns:
			dataobject (dict) 
	'''
	if config.mode == 'train':
		logger.debug('Loading Training Data...')

		'''Load Datasets'''
		if config.lang == 'Dyck' and config.generate:
			## Code for generating the Dyck  language dataset.
			train_corpus 	= DyckCorpus(config.p_val, config.q_val, config.num_par, config.lower_window, config.upper_window, config.training_size, config.lower_depth, config.upper_depth, config.debug)
			val_corpus = DyckCorpus(config.p_val, config.q_val, config.num_par, config.lower_window, config.upper_window, config.test_size, config.lower_depth, config.upper_depth, config.debug)
			if config.generalize:
				val_gen_corpus 	= DyckCorpus(config.p_val, config.q_val, config.num_par, config.val_lower_window, config.val_upper_window, config.test_size, config.lower_depth, config.upper_depth, config.debug)
				val_corpus_bins = [val_corpus, val_gen_corpus]
			else:
				val_corpus_bins = [val_corpus]
		elif config.lang == 'Counter' and config.generate:
			train_corpus 	= CounterCorpus( config.num_par, config.lower_window, config.upper_window, config.training_size, config.debug)
			val_corpus 	= CounterCorpus( config.num_par, config.lower_window, config.upper_window, config.test_size, config.debug)
			val_corpus_bins = [val_corpus]

		else:
			# Load previously generated data.
			data_dir = os.path.join(data_path, config.dataset)
			with open(os.path.join(data_dir, 'train_corpus.pk'), 'rb') as f:
				## The corpus contains the raw sequences and targets for Training
				train_corpus = pickle.load(f)
			## Inspect what raw sequences looks like.
			# print(train_corpus.source)
			# print(train_corpus.target)
			with open(os.path.join(data_dir, 'val_corpus_bins.pk'), 'rb') as f:
				val_corpus_bins = pickle.load(f)
		voc = Voc()
		"""
		create_vocab_dict(train_corpus) scans all training strings, extracts the unique symbols (e.g. a, b, parentheses), and builds two maps:
		word2index: token → integer index; index2word: integer index → token
		"""
		voc.create_vocab_dict(train_corpus)
		voc.noutputs = train_corpus.noutputs
		"""
		Sampler:Takes raw strings and uses voc.word2index to convert them into PyTorch tensors of token indices.
		1. train_loader:Batches of training examples, where each example is a pair
		input_tensor: a 1D (or 2D if batched) tensor of token indices, e.g. [3, 5, 2, 2, 7].
		label_tensor: a tensor of target labels (e.g. a single integer 0/1 for accept/reject).
		Iterating over train_loader yields (input_batch, label_batch) tuples.
		=====================
		val_loader_bins: A list of Sampler objects, each element is a validation loader corresponding to one validation “bin”.
		Like train_loader, each loader yields (input_batch, label_batch) tuples.
		"""
		train_loader = Sampler(train_corpus, voc, config.batch_size, config.bptt)
		val_loader_bins = [Sampler(val_corpus_bin, voc, config.batch_size, config.bptt) for val_corpus_bin in val_corpus_bins]
		msg = 'Training and Validation Data Loaded:\nTrain Size: {}'.format(len(train_loader))
		logger.info(msg)

		return train_loader, val_loader_bins, voc

	else:	## For testing.
		logger.debug('Loading Test Data...')

		'''Load Datasets'''
		if config.lang == 'Dyck':
			if config.generate:
				test_corpus = DyckCorpus(config.p_val, config.q_val, config.num_par, config.lower_window, config.upper_window, config.test_size, config.lower_depth, config.upper_depth, config.debug)
				if config.generalize:
					test_gen_corpus 	= DyckCorpus(config.p_val, config.q_val, config.num_par, config.test_lower_window, config.test_upper_window, config.test_size, config.lower_depth, config.upper_depth, config.debug)
					test_corpus_bins = [test_corpus, test_gen_corpus]
				else:
					test_corpus_bins = [test_corpus]
			else:
				data_dir = os.path.join(data_path, config.dataset)
				with open(os.path.join(data_dir, 'val_corpus_bins.pk'), 'rb') as f:
					test_corpus_bins = pickle.load(f)

		elif config.lang == 'Counter':
			test_corpus 	= CounterCorpus( config.num_par, config.lower_window, config.upper_window, config.test_size, config.debug)
			test_corpus_bins = [test_corpus]

		test_loader_bins = [Sampler(test_corpus_bin, voc, config.batch_size, config.bptt) for test_corpus_bin in test_corpus_bins]

		msg = 'Test Data Loaded:\n'
		logger.info(msg)


		return test_loader_bins


def main():
	'''read arguments'''
	parser = build_parser() ## build parser is handled in args.py
	args = parser.parse_args()
	config =args
	mode = config.mode
	if mode == 'train':
		is_train = True
	else:
		is_train = False

	''' Set seed for reproducibility'''
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	random.seed(config.seed)

	'''GPU initialization'''
	device = gpu_init_pytorch(config.gpu)
	'''Run Config files/paths'''
	run_name = config.run_name
	config.log_path = os.path.join(log_folder, run_name)
	config.model_path = os.path.join(model_folder, run_name)
	config.board_path = os.path.join(board_path, run_name)

	vocab_path = os.path.join(config.model_path, 'vocab.p')
	config_file = os.path.join(config.model_path, 'config.p')
	log_file = os.path.join(config.log_path, 'log.txt')

	if config.results:
		config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

	if is_train:
		create_save_directories(config.log_path, config.model_path)
	else:
		create_save_directories(config.log_path, config.result_path)

	logger = get_logger(run_name, log_file, logging.DEBUG)
	writer = SummaryWriter(config.board_path)

	logger.debug('Created Relevant Directories')
	logger.info('Experiment Name: {}'.format(config.run_name))

	'''Read Files and create/load Vocab'''
	if is_train:
		logger.debug('Creating Vocab and loading Data ...')
		############################
		train_loader, val_loader_bins, voc  = load_data(config, logger)
		logger.info('Vocab Created with number of words : {}'.format(voc.nwords))
		# print(train_loader.get_batch(1))
		with open(vocab_path, 'wb') as f:
			pickle.dump(voc, f, protocol=pickle.HIGHEST_PROTOCOL)
		logger.info('Vocab saved at {}'.format(vocab_path))
	else:
		logger.info('Loading Vocab File...')

		with open(vocab_path, 'rb') as f:
			voc = pickle.load(f)

		logger.info('Vocab Files loaded from {}'.format(vocab_path))

		logger.info("Loading Test Dataloaders...")
		config.batch_size = 1
		test_loader_bins = load_data(config, logger, voc)
		logger.info("Done loading test dataloaders")

	if is_train:
		max_val_acc = 0.0
		epoch_offset= 0
		## Build and initialize the model.
		if config.load_model:
			checkpoint = get_latest_checkpoint(config.model_path, logger)
			if checkpoint:
				ckpt = torch.load(checkpoint, map_location=lambda storage, loc: storage)
				#config.lr = checkpoint['lr']
				model = build_model(config=config, voc=voc, device=device, logger=logger)
				model.load_state_dict(ckpt['model_state_dict'])
				model.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
		else:
			model = build_model(config=config, voc=voc, device=device, logger=logger)
		# pdb.set_trace()
		## Start model Training.
		logger.info('Initialized Model')
		with open(config_file, 'wb') as f:
			pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)
		logger.debug('Config File Saved')
		logger.info('Starting Training Procedure')
		train_model(model, train_loader, val_loader_bins, voc,
					device, config, logger, epoch_offset, max_val_acc, writer)
	else:
		gpu = config.gpu
		with open(config_file, 'rb') as f:
			bias = config.bias
			extraffn = config.extraffn
			config = AttrDict(pickle.load(f))
			config.gpu = gpu
			config.bins = len(test_loader_bins)
			config.batch_size = 1
			config.bias = bias
			config.extraffn = extraffn
			# To do: remove it later
			#config.num_labels =2  

		model = build_model(config=config, voc=voc, device=device, logger=logger)
		checkpoint = get_latest_checkpoint(config.model_path, logger)
		ep_offset, train_loss, score, voc = load_checkpoint(
			model, config.mode, checkpoint, logger, device, bins = config.bins)

		logger.info('Prediction from')
		od = OrderedDict()
		od['epoch'] = ep_offset
		od['train_loss'] = train_loss
		if config.bins != -1:
			for i in range(config.bins):
				od['max_val_acc_bin{}'.format(i)] = score[i]
		else:
			od['max_val_acc'] = score
		print_log(logger, od)
		pdb.set_trace()
		#test_acc_epoch, test_loss_epoch = run_validation(config, model, test_loader, voc, device, logger)
		#test_analysis_dfs = []
		for i in range(config.bins):
			test_acc_epoch, test_analysis_df = run_test(config, model, test_loader_bins[i], voc, device, logger)
			logger.info('Bin {} Accuracy: {}'.format(i, test_acc_epoch))
			#test_analysis_dfs.append(test_analysis_df)
			test_analysis_df.to_csv(os.path.join(result_folder, '{}_{}_test_analysis_bin{}.csv'.format(config.dataset, config.model_type, i)))
		logger.info("Analysis results written to {}...".format(result_folder))

if __name__ == '__main__':
	main()