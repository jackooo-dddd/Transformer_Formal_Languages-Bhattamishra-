import os
import sys
import math
import logging
import random
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from gensim import models

from src.components.rnns import RNNModel
from src.components.transformers import TransformerModel, TransformerXLModel, SimpleTransformerModel
from src.components.mogrifierLSTM import MogrifierLSTMModel
# from src.components.sa_rnn import SARNNModel

from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint

from src.visualize_san import generate_visualizations

import ipdb as pdb

from collections import OrderedDict
import copy


class LanguageModel(nn.Module):
	def __init__(self, config, voc, device, logger):
		super(LanguageModel, self).__init__()

		self.config = config
		self.device = device
		self.logger= logger
		self.voc = voc
		self.lr =config.lr
		self.epsilon = 0.5

		self.logger.debug('Initalizing Model...')
		self._initialize_model()

		self.logger.debug('Initalizing Optimizer and Criterion...')
		self._initialize_optimizer()

		# nn.CrossEntropyLoss() does both F.log_softmax() and nn.NLLLoss()
		# self.criterion = nn.NLLLoss()
		# self.criterion = nn.CrossEntropyLoss(reduction= 'none')
		self.criterion = nn.MSELoss(reduction = 'none')

	def _initialize_model(self):
		if self.config.model_type == 'RNN':
			self.model = RNNModel(self.config.cell_type, self.voc.nwords, self.voc.noutputs, self.config.emb_size, self.config.hidden_size, self.config.depth, self.config.dropout, self.config.tied, self.config.use_emb).to(self.device)
		elif self.config.model_type == 'SAN':
			self.model = TransformerModel(self.voc.nwords, self.voc.noutputs,
										self.config.d_model, self.config.heads,
										self.config.d_ffn, self.config.depth,
										self.config.dropout, pos_encode = self.config.pos_encode,
										bias = self.config.bias, pos_encode_type= self.config.pos_encode_type,
										max_period = self.config.max_period).to(self.device)

		elif self.config.model_type == 'SAN-Simple':
			self.model = SimpleTransformerModel(self.voc.nwords, self.voc.noutputs, self.config.d_model,
												self.config.heads, self.config.d_ffn, self.config.depth,
												self.config.dropout, pos_encode = self.config.pos_encode, bias = self.config.bias,
												posffn= self.config.posffn, freeze_emb= self.config.freeze_emb,
												freeze_q = self.config.freeze_q, freeze_k = self.config.freeze_k,
												freeze_v = self.config.freeze_v, freeze_f = self.config.freeze_f,
												zero_keys = self.config.zero_k, pos_encode_type= self.config.pos_encode_type,
												max_period = self.config.max_period).to(self.device)
		elif self.config.model_type == 'SAN-Rel':
			self.model = TransformerXLModel(self.voc.nwords, self.voc.noutputs, self.config.d_model, self.config.heads, self.config.d_ffn, self.config.depth, self.config.dropout).to(self.device)
		elif self.config.model_type == 'Mogrify':
			self.model = MogrifierLSTMModel(self.config.cell_type, self.voc.nwords, self.config.emb_size, self.config.hidden_size, self.config.depth, self.config.dropout, self.config.tied).to(self.device)
		elif self.config.model_type == 'SARNN':
			self.model = SARNNModel(self.config.cell_type, self.voc.nwords, self.config.emb_size, self.config.hidden_size, self.config.depth, self.config.dropout, self.config.tied).to(self.device)


	def _initialize_optimizer(self):
		self.params = self.model.parameters()

		if self.config.opt == 'adam':
			self.optimizer = optim.Adam(self.params, lr=self.config.lr)
		elif self.config.opt == 'adadelta':
			self.optimizer = optim.Adadelta(self.params, lr=self.config.lr)
		elif self.config.opt == 'asgd':
			self.optimizer = optim.ASGD(self.params, lr=self.config.lr)
		elif self.config.opt =='rmsprop':
			self.optimizer = optim.RMSprop(self.params, lr=self.config.lr)
		else:
			self.optimizer = optim.SGD(self.params, lr=self.config.lr)
			self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=self.config.decay_rate, patience=self.config.decay_patience, verbose=True)


	def trainer(self, source, targets, lengths, hidden, config, device=None ,logger=None):
		"""
		   Perform one training step (forward, backward, parameter update) on a single batch.
		   Args:
		       source (Tensor): input token indices of shape (seq_len, batch_size)
		       targets (Tensor): one‑hot (or probability) target vectors of shape (seq_len, batch_size, n_outputs).
		       lengths (Tensor): actual lengths of each sequence in the batch, shape (batch_size,).
		       hidden (Tensor or tuple): previous hidden state for RNN‑style models (None for Transformer).
		       config (argparse.Namespace): training configuration, including model_type, max_grad_norm, etc.
		   Returns:
		       loss_value (float): the average loss over non‑padding tokens in this batch.
		       new_hidden (Tensor or tuple): the updated hidden state (for RNN variants) or None (for Transformer).
		   """
		self.optimizer.zero_grad() # Resets all accumulated gradients
		if config.model_type == 'RNN':
			output, hidden = self.model(source, hidden, lengths)
		elif config.model_type == 'SAN' or config.model_type == 'SAN-Rel' or config.model_type == 'SAN-Simple':
			## Call the Transformer which returns only output (no recurrent hidden state).
			## self.model(source) produces raw scores (logits) for every token in the batch.
			## The line where Transformer actually “predicts.”
			"""
			output = self.model(source)
			Internally, PyTorch:
			Checks that self.model is a subclass of nn.Module (which it is — e.g. TransformerModel).
			Calling a nn.Module instance executes its forward() method.
			Applies pre-hooks (if any), Runs self.forward(source), Returns the result.
			(Powerful<=>can swap architectures by changing one config flag and everything else still works)
			"""
			# print("MyFlag1")
			output = self.model(source)
		elif config.model_type == 'Mogrify':
			output, hidden = self.model(source, hidden)
		elif config.model_type == 'SARNN':
			output, hidden = self.model(source, hidden)
		mask = (source != 0).float().reshape(-1)
		## Compute the lost per token.
		loss = self.criterion(output.view(-1,self.voc.noutputs), targets.contiguous().view(-1,self.voc.noutputs)).mean(1)
		loss = (mask * loss).sum()/ mask.sum()
		loss.backward() ## Backpropagate Computes gradients of the loss w.r.t. all model parameters.
		if self.config.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
		self.optimizer.step() ## Here the optimizer reads each parameter’s .grad and applies its update rule
		# for p in self.model.parameters():
			# p.data.add_(-self.lr, p.grad.data)
		if config.model_type != 'SAN' and config.model_type != 'SAN-Simple' and config.model_type != 'SAN-Rel':
			hidden = self.repackage_hidden(hidden)
		return loss.item(), hidden  ## hidden state is None for the Transformer model.

	def evaluator(self, source, targets, lengths, hidden, config, device=None ,logger=None):
		"""
		    Run a forward pass on one batch in evaluation mode and compute batch accuracy.
			Called inside your validation loop to measure how well the model generalizes in the validation bins.
		    Args:
		        source (Tensor): input token indices, shape (seq_len, batch_size).
		        targets (Tensor): one‑hot (or probability) targets, shape (seq_len, batch_size, n_outputs).
		        lengths (Tensor): actual lengths of each sequence in the batch.
		        hidden (Tensor or tuple): previous hidden state for RNN models (None for Transformers).
		        config (Namespace): configuration, including model_type and epsilon threshold.
		    Returns:
		        batch_acc (float): fraction of sequences in this batch perfectly predicted.
		        new_hidden (Tensor or tuple): updated hidden state (for RNN variants) or None (for Transformers).
		    """
		## Get raw outputs from the model.
		# print("source", source, source.shape) ## [Sequence length, 32]
		# print("targets", targets, targets.shape) ## [Sequence length, 32, 2]
		"""
		MY understanding:
		source[t, b] is the token at position t in example b.(model sees the t‑th character)
 		targets[t, b] is the token that follows position t in example b*—i.e. the “next character.”
 		Because in autoregressive character‑prediction task, we present the model with the prefix of a string up to 
 		character t, and ask it to predict the character at position t + 1.
 		There is a "One step shift" from the source to the target.
 		See dataloader.getBatch() of how the source and targets are generated.
		"""
		if config.model_type == 'RNN':
			output, hidden = self.model(source, hidden, lengths)
		elif config.model_type == 'SAN' or config.model_type == 'SAN-Rel' or config.model_type == 'SAN-Simple':
			output = self.model(source)
		elif config.model_type == 'Mogrify':
			output, hidden = self.model(source, hidden)
		elif config.model_type == 'SARNN':
			output, hidden = self.model(source, hidden)
		batch_acc = 0.0
		mask = (source!=0).float().unsqueeze(-1) ## mask to zero out padding tokens
		masked_output = mask*output
		try:
			## Threshold model probabilities into binary predictions/ Compare against self.epsilon (e.g. 0.5)
			# Each row t of out_j is the one‑hot (or binary) prediction for timestep t,
			# and each row of target_j is the true one‑hot target for timestep t.
			out_np= np.int_(masked_output.detach().cpu().numpy() >= self.epsilon)
			target_np = np.int_(targets.detach().cpu().numpy())
		except:
			pdb.set_trace()
		# print("TARGET_NP", target_np, target_np.shape)
		# # print("OUTPUT", output,output.shape)
		# print("*******************")
		# print("OUTPUT_NP", out_np, out_np.shape)
		"""
		torch.Size([198, 32, 2]
		198 is the sequence length, 32 is the batch size.
		2 is the vocabulary size, meaning we have only 2 possible characters.
		[1, 1] means both of those probabilities exceeded 0.5. 
		From the targets:
		[1, 1]   # both tokens would keep you in the language
		[1, 0]   # only token 0 is valid here
		[0, 1]   # only token 1 is valid here
		[0, 0]   # neither token keeps you in the language
		"""
		## For each sequence in the batch, checks whether all predicted tokens exactly
		# match the target tokens (target_np). If they do, increments batch_acc by 1.
		for j in range(out_np.shape[1]):
			out_j = out_np[:,j] ## shape: (seq_len, n_outputs)
			target_j = target_np[:,j] ## shape: (seq_len, n_outputs)
			## Check if *all* elements at position j matches with the target j.
			# np.all or (==).all() checks elementwise equality across the sequence
			if np.all(np.equal(out_j, target_j)) and (out_j.flatten() == target_j.flatten()).all():
				batch_acc+=1 # If so, set `pred` as one
		batch_acc = batch_acc/source.size(1)
		if config.model_type != 'SAN' and config.model_type != 'SAN-Simple' and config.model_type != 'SAN-Rel':
			hidden = self.repackage_hidden(hidden) ## Detach hidden state for RNNs to avoid backprop through eval
		return  batch_acc, hidden


	def repackage_hidden(self, h):
		"""Wraps hidden states in new Tensors, to detach them from their history."""
		if isinstance(h, torch.Tensor):
			return h.detach()
		else:
			return tuple(self.repackage_hidden(v) for v in h)


def build_model(config, voc, device, logger):
	'''
		Add Docstring
	'''
	model = LanguageModel(config, voc, device, logger)
	model = model.to(device)

	return model



def train_model(model, train_loader, val_loader_bins, voc, device, config, logger, epoch_offset= 0, max_val_acc=0.0, writer= None):
	"""
    Process the full training loop for a language model with periodic validation, checkpointing, and TensorBoard logging.
    Args:
        model: the wrapped PyTorch model (with .trainer and .evaluator methods).
        train_loader (Sampler): yields training batches via get_batch(i).
        val_loader_bins (List[Sampler]): one or more validation loaders and (optionally) generalization tests.
        voc (Voc): vocabulary object containing word2index, id2word, nwords, noutputs.
        config: all hyperparameters and flags, including:
            - epochs (int): total number of epochs to train.
            - batch_size (int): number of examples per batch.
            - model_type (str): e.g. 'RNN', 'SAN', etc.
            - lr, opt, max_grad_norm, checkpoint_interval, epsilon, etc.
        epoch_offset (int, optional): starting epoch index (used when resuming from checkpoint).
        max_val_acc (float, optional): best validation accuracy seen so far.
        writer (SummaryWriter, optional): TensorBoard writer for scalar/graph logging.
    Side effects:
        - Trains the model in-place, updating its parameters.
        - At each epoch:
            • Runs validation on each val_loader_bin.
            • Saves a new checkpoint if any validation accuracy exceeds max_val_acc.
            • Logs losses and accuracies to console and TensorBoard.
    """
	if config.histogram and writer:
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, epoch_offset)

	estop_count=0
	n_bins = len(val_loader_bins) ## Number of different validation bins
	max_train_acc = max_val_acc
	best_train_epoch = 0 ## Track the epoch at which each bin achieved its best accuracy
	max_val_acc_bins = [max_val_acc for i in range(n_bins)]
	best_epoch_bins = [0 for i in range(n_bins)]
	iters = 0
	viz_every = int(train_loader.num_batches // 4)
	try:
		for epoch in range(1, config.epochs + 1):
			od = OrderedDict()
			od['Epoch'] = epoch + epoch_offset
			print_log(logger, od)

			train_loss_epoch = 0.0
			train_acc_epoch = 0.0
			val_acc_epoch = 0.0

			# Train Mode
			model.train()
			start_time= time()
			# Batch-wise Training

			lr_epoch =  model.optimizer.state_dict()['param_groups'][0]['lr'] ## Get current learning rate
			for batch, i in enumerate(range(0, len(train_loader), config.batch_size)): # Batch-wise training loop (Inside some training epoch)
				if config.viz and config.model_type == 'SAN' and batch % viz_every == 0:  # Optionally generate visualizations for SAN models
					val_acc_bins = [run_validation(config, model, val_loader_bins[i], voc, device, logger) for i in range(1,4)]
					generate_visualizations(model, config, voc, run_name = config.run_name, iteration = iters, score = sum(val_acc_bins)/3, device = device)
					if writer:
						bin_acc_dict = {'bin{}_score'.format(i+1) : val_acc_bins[i] for i in range(3)}
						writer.add_scalars('acc/val_acc_iters', bin_acc_dict, iters)
				# Initialize hidden state for RNNs, None for Transformers
				if config.model_type != 'SAN' and config.model_type != 'SAN-Simple' and config.model_type != 'SAN-Rel':
					hidden = model.model.init_hidden(config.batch_size)
				else:
					hidden = None
				# Load a batch of Training data.
				source, targets, word_lens = train_loader.get_batch(i)
				source, targets, word_lens = source.to(device), targets.to(device), word_lens.to(device)

				# Perform a training step: forward, backward, update(Happens in model.trainer())
				loss, hidden = model.trainer(source, targets, word_lens, hidden, config)
				train_loss_epoch += loss #* len(source)
				iters += 1
			## Average training loss over all batches
			train_loss_epoch = train_loss_epoch / train_loader.num_batches
			time_taken = (time() - start_time)/60.0

			if writer: ## Log training loss to TensorBoard if write.
				writer.add_scalar('loss/train_loss', train_loss_epoch, epoch + epoch_offset)

			logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))
			logger.debug('Starting Validation')
			# Run validation on each bin
			val_acc_epoch_bins = [run_validation(config, model, val_loader_bin, voc, device, logger) for val_loader_bin in val_loader_bins]
			train_acc_epoch = run_validation(config, model, train_loader, voc, device, logger)
			if train_acc_epoch ==  max(max_train_acc, train_acc_epoch):  ## Update best training accuracy and epoch
				best_train_epoch = epoch
				max_train_acc = train_acc_epoch

			val_acc_epoch = np.mean(val_acc_epoch_bins)
			if config.opt == 'sgd' and model.scheduler: # Step learning-rate scheduler if using SGD
				model.scheduler.step(val_acc_epoch)

			save_flag = False
			for i in range(n_bins): # Check if validation improved in any bin
				if val_acc_epoch_bins[i] > max_val_acc_bins[i]:
					save_flag = True
					max_val_acc_bins[i] = val_acc_epoch_bins[i]
					#max_train_acc = train_acc_epoch
					best_epoch_bins[i] = epoch

					logger.debug('Validation Accuracy bin{} : {}'.format(i, val_acc_epoch_bins[i]))
			if save_flag: ## Save a checkpoint if needed
				state = {
							'epoch' : epoch + epoch_offset,
							'model_state_dict': model.state_dict(),
							'voc': model.voc,
							'optimizer_state_dict': model.optimizer.state_dict(),
							'train_loss' : train_loss_epoch,
							'lr' : lr_epoch
						}
				for i in range(n_bins):
					state['val_acc_epoch_bin{}'.format(i)] = val_acc_epoch_bins[i]
					state['max_val_acc_bin{}'.format(i)] = max_val_acc_bins[i]

				save_checkpoint(state, epoch + epoch_offset, logger, config.model_path, config.ckpt)

			if writer: ## Log scalars to TensorBoard: training & validation accuracies
				writer.add_scalar('acc/train_acc', train_acc_epoch, epoch + epoch_offset)
				for i in range(n_bins):
					writer.add_scalar('acc/val_acc_bin{}'.format(i), val_acc_epoch_bins[i], epoch + epoch_offset)
				#writer.add_scalar('acc/val_gen_acc', val_gen_acc_epoch, epoch + epoch_offset)

			## Summary logging for this epoch
			od = OrderedDict()
			od['Epoch'] = epoch + epoch_offset
			od['train_loss'] = train_loss_epoch
			od['train_acc'] = train_acc_epoch
			od['lr_epoch'] = lr_epoch
			for i in range(n_bins):
				od['val_acc_epoch_bin{}'.format(i)] = val_acc_epoch_bins[i]
				od['max_val_acc_bin{}'.format(i)] = max_val_acc_bins[i]

			print_log(logger, od)
			## Optionally log parameter histograms again at epoch end
			if config.histogram and writer:
				for name, param in model.named_parameters():
					writer.add_histogram(name, param, epoch + epoch_offset)
			## Enable early stopings.
			# if estop_count > 10:
			# 	logger.debug('Early Stopping at Epoch: {} after no improvement in {} epochs'.format(epoch, estop_count))
			# 	break
			if np.mean(val_acc_epoch_bins) >= 0.999:
				logger.info('Reached optimum performance!')
				break

		writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars.json'))
		writer.close()
		logger.info('Training Completed for {} epochs'.format(config.epochs))
		if config.results:
			store_results(config, max_val_acc_bins, max_train_acc, best_train_epoch, train_loss_epoch, best_epoch_bins)
			logger.info('Scores saved at {}'.format(config.result_path))

	except KeyboardInterrupt:
		logger.info('Exiting Early....')
		if config.results:
			store_results(config, max_val_acc_bins, max_train_acc, best_train_epoch, train_loss_epoch, best_epoch_bins)
			logger.info('Scores saved at {}'.format(config.result_path))



def run_validation(config, model, val_loader, voc, device, logger):
	"""
	    Evaluate the model’s sequence‑level accuracy over an entire validation loader.
	    Args:
	        config (Namespace): configuration, including model_type and batch_size.
	        model (LanguageModel): the model instance with .evaluator method.
	        val_loader (Sampler): yields validation batches via get_batch(i).
	        voc (Voc): vocabulary object (not used directly here but kept for consistency).
	    Returns: the average sequence‑level accuracy over all batches.
	    """
	batch_num = 0
	val_acc_epoch =0.0
	model.eval() ## Switch to evaluation mode
	with torch.no_grad(): ## Disable gradient calculations for speed and memory
		## Iterate over the validation set
		for batch, i in enumerate(range(0, len(val_loader), val_loader.batch_size)):
			if config.model_type != 'SAN' and config.model_type != 'SAN-Simple' and config.model_type != 'SAN-Rel':
				hidden = model.model.init_hidden(config.batch_size) ## Initialize hidden state for RNN variants
			else:
				hidden = None
			## Fetch this batch: source [seq_len, batch], targets [seq_len, batch, n_outputs], lengths
			source, targets, word_lens = val_loader.get_batch(i)
			source, targets, word_lens = source.to(device), targets.to(device), word_lens.to(device)
			## Use the model’s evaluator to get sequence‑level accuracy for this batch
			acc, hidden = model.evaluator(source, targets, word_lens, hidden, config)
			val_acc_epoch += acc
			batch_num += 1

	val_acc_epoch = val_acc_epoch / val_loader.num_batches # Compute the mean accuracy across all batches

	return val_acc_epoch

def run_test(config, model, test_loader, voc, device, logger):
	batch_num =1
	test_acc_epoch =0.0
	strings = []
	correct_or_not = []
	lengths = []
	depths = []
	model.eval()

	with torch.no_grad():
		for batch, i in enumerate(range(len(test_loader) - 1)):
			if config.model_type != 'SAN' and config.model_type != 'SAN-Simple' and config.model_type != 'SAN-Rel':
				hidden = model.model.init_hidden(config.batch_size)
			else:
				hidden = None
			try:
				source, targets, word_lens = test_loader.get_batch(i)
			except Exception as e:
				pdb.set_trace()
			source, targets, word_lens = source.to(device), targets.to(device), word_lens.to(device)
			acc, hidden = model.evaluator(source, targets, word_lens, hidden, config)
			test_acc_epoch += acc

			source_str = test_loader.data[i]
			source_len = len(source_str)
			source_depth = test_loader.Lang.depth_counter(source_str).sum(1).max()
			strings.append(source_str)
			lengths.append(source_len)
			depths.append(source_depth)
			correct_or_not.append(acc)

			print("Completed {}/{}...".format(i+1, len(test_loader) - 1), end = '\r', flush = True)

	test_acc_epoch = test_acc_epoch / (len(test_loader) - 1)
	test_analysis_df = pd.DataFrame(
						{
							'String' : strings,
							'Length' : lengths,
							'Depth'  : depths,
							'Score'	 : correct_or_not
						}
					)

	return test_acc_epoch, test_analysis_df


