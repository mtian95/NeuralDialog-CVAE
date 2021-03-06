#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import os
import time
import sys

import numpy as np
import tensorflow as tf
from beeprint import pp

from config_utils import KgCVAEConfig as Config
from config_utils import RNNBaselineConfig as RNNConfig
from data_apis.corpus import SWDADialogCorpus
from data_apis.data_utils import SWDADataLoader, RNNDataLoader
from models.cvae import KgRnnCVAE
from models.hier_baseline import HierBaseline
from ldamodel import LDAModel

# constants
tf.app.flags.DEFINE_string("model_type", None, "The type of model to train. Can be kgcvae, hierbaseline, rnnbaseline")
tf.app.flags.DEFINE_string("word2vec_path", None, "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("data_dir", "data/dbpedia_small.p", "Raw data directory.") # TODO redirect this to the correct corpus
tf.app.flags.DEFINE_string("work_dir", "working", "Experiment results directory.")
tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_string("test_path", "run1500783422", "the dir to load checkpoint for forward only")
tf.app.flags.DEFINE_string("lda_model_path", "lda/lda_model", "the path to pretrained LDA model")
tf.app.flags.DEFINE_string("id2word_path", "lda/id2word_wiki.txt", "the path to the id2word dict for LDA model")
tf.app.flags.DEFINE_string("vocab_dict_path", "data/vocab", "Vocab files directory.")
tf.app.flags.DEFINE_bool("use_imdb", False, "whether to use the keras imdb dataset")

FLAGS = tf.app.flags.FLAGS


def main(model_type):
	# config for training
	config = Config()

	# config for validation
	valid_config = Config()
	valid_config.keep_prob = 1.0
	valid_config.dec_keep_prob = 1.0
	valid_config.batch_size = 60

	# configuration for testing
	test_config = Config()
	test_config.keep_prob = 1.0
	test_config.dec_keep_prob = 1.0
	test_config.batch_size = 1

	pp(config)

	# which model to run
	if model_type == "kgcvae":
		model_class = KgRnnCVAE
		backward_size = config.backward_size
		config.use_hcf, valid_config.use_hcf, test_config.use_hcf = True
	elif model_type == "cvae":
		model_class = KgRnnCVAE
		backward_size = config.backward_size
		config.use_hcf, valid_config.use_hcf, test_config.use_hcf = False
	elif model_type == 'hierbaseline':
		model_class = HierBaseline
		backward_size = config.backward_size
	else:
		raise ValueError("This shouldn't happen.")

	# LDA Model
	ldamodel = LDAModel(config, trained_model_path=FLAGS.lda_model_path, id2word_path=FLAGS.id2word_path)

	# get data set
	api = SWDADialogCorpus(FLAGS.data_dir, word2vec=FLAGS.word2vec_path, word2vec_dim=config.embed_size, vocab_dict_path=FLAGS.vocab_dict_path,
							lda_model=ldamodel, imdb=FLAGS.use_imdb)

	dial_corpus = api.get_dialog_corpus()
	meta_corpus = api.get_meta_corpus()

	train_meta, valid_meta, test_meta = meta_corpus.get("train"), meta_corpus.get("valid"), meta_corpus.get("test")
	train_dial, valid_dial, test_dial = dial_corpus.get("train"), dial_corpus.get("valid"), dial_corpus.get("test")

	# convert to numeric input outputs that fits into TF models
	train_feed = SWDADataLoader("Train", train_dial, train_meta, config)
	valid_feed = SWDADataLoader("Valid", valid_dial, valid_meta, config)
	test_feed = SWDADataLoader("Test", test_dial, test_meta, config)

	# if you're testing an existing implementation or resuming training
	if FLAGS.forward_only or FLAGS.resume: 
		log_dir = os.path.join(FLAGS.work_dir+"_"+FLAGS.model_type, FLAGS.test_path)
	else:
		log_dir = os.path.join(FLAGS.work_dir+"_"+FLAGS.model_type, "run"+str(int(time.time())))

	# begin training
	with tf.Session() as sess:
		initializer = tf.random_uniform_initializer(-1.0 * config.init_w, config.init_w)
		scope = "model"
		with tf.variable_scope(scope, reuse=None, initializer=initializer):
			model = model_class(sess, config, api, log_dir=None if FLAGS.forward_only else log_dir, forward=False, scope=scope)
		with tf.variable_scope(scope, reuse=True, initializer=initializer):
			valid_model = model_class(sess, valid_config, api, log_dir=None, forward=False, scope=scope)
		with tf.variable_scope(scope, reuse=True, initializer=initializer):
			test_model = model_class(sess, test_config, api, log_dir=None, forward=True, scope=scope)

		print("Created computation graphs")
		if api.word2vec is not None and not FLAGS.forward_only:
			print("Loaded word2vec")
			sess.run(model.embedding.assign(np.array(api.word2vec)))

		# write config to a file for logging
		if not FLAGS.forward_only:
			with open(os.path.join(log_dir, "run.log"), "wb") as f:
				f.write(pp(config, output=False))

		# create a folder by force
		ckp_dir = os.path.join(log_dir, "checkpoints")
		if not os.path.exists(ckp_dir):
			os.mkdir(ckp_dir)

		ckpt = tf.train.get_checkpoint_state(ckp_dir)
		print("Created models with fresh parameters.")
		sess.run(tf.global_variables_initializer())

		if ckpt:
			print("Reading dm models parameters from %s" % ckpt.model_checkpoint_path)
			model.saver.restore(sess, ckpt.model_checkpoint_path)

		# if you're training a model
		if not FLAGS.forward_only: 

			dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__+ ".ckpt")
			global_t = 1
			patience = 10  # wait for at least 10 epoch before stop
			dev_loss_threshold = np.inf
			best_dev_loss = np.inf


			# train for a max of max_epoch's. saves the model after the epoch if it's some amount better than current best
			for epoch in range(config.max_epoch):
				print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))

				# begin training
				if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
					train_feed.epoch_init(config.batch_size, backward_size,
										  config.step_size, shuffle=True)

				global_t, train_loss = model.train(global_t, sess, train_feed, update_limit=config.update_limit)

				# begin validation and testing
				valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
									  valid_config.step_size, shuffle=False, intra_shuffle=False)
				valid_loss = valid_model.valid("ELBO_VALID", sess, valid_feed)

				test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
									 test_config.step_size, shuffle=True, intra_shuffle=False)
				test_model.test(sess, test_feed, num_batch=1) #TODO change this batch size back to a reasonably large number


				done_epoch = epoch + 1
				# only save a models if the dev loss is smaller
				# Decrease learning rate if no improvement was seen over last 3 times.
				if config.op == "sgd" and done_epoch > config.lr_hold:
					sess.run(model.learning_rate_decay_op)

				if True: #valid_loss < best_dev_loss: # TODO this change makes the model always save. Change this back when corpus not trivial
					if True: #valid_loss <= dev_loss_threshold * config.improve_threshold:
						patience = max(patience, done_epoch * config.patient_increase)
						# dev_loss_threshold = valid_loss

					# still save the best train model
					if FLAGS.save_model:
						print("Save model!!")
						model.saver.save(sess, dm_checkpoint_path, global_step=epoch)
					# best_dev_loss = valid_loss

				if config.early_stop and patience <= done_epoch:
					print("!!Early stop due to run out of patience!!")
					break
			# print("Best validation loss %f" % best_dev_loss)
			print("Done training")


		# else if you're just testing an existing model
		else:
			# begin validation
			valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
								  valid_config.step_size, shuffle=False, intra_shuffle=False)
			valid_model.valid("ELBO_VALID", sess, valid_feed)

			test_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
								  valid_config.step_size, shuffle=False, intra_shuffle=False)
			valid_model.valid("ELBO_TEST", sess, test_feed)


			# begin testing
			dest_f = open(os.path.join(log_dir, "test.txt"), "wb")
			test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
								 test_config.step_size, shuffle=False, intra_shuffle=False)
			test_model.test(sess, test_feed, num_batch=None, repeat=10, dest=dest_f)
			dest_f.close()


if __name__ == "__main__":
	FLAGS = tf.app.flags.FLAGS

	if FLAGS.forward_only:
		if FLAGS.test_path is None:
			print("Set test_path before forward only")
			exit(1)

	# check if model_type was specified
	if FLAGS.model_type is None:
		raise ValueError("Must give a model_type (kgcvae, cvae, hierbaseline)")
	elif FLAGS.model_type.lower().strip() == "hierbaseline" or FLAGS.model_type.lower().strip() == "kgcvae" or FLAGS.model_type.lower().strip() == "cvae":
		model_type = FLAGS.model_type.lower().strip()
		main(model_type)
	else:
		raise ValueError("model_type must be kgcvae, hierbaseline, or rnnbaseline")













