import os
import time
import sys

import numpy as np
import tensorflow as tf
from beeprint import pp

from config_utils import KgCVAEConfig as Config
from data_apis.corpus import SWDADialogCorpus
from data_apis.data_utils import SWDADataLoader
from models.hier_baseline import HierBaseline
from ldamodel import LDAModel

from kgcvae_swda import main

# TODO make these flags not be connected to kgcvae_swda
# tf.app.flags.DEFINE_string("word2vec_path", None, "The path to word2vec. Can be None.")
# # tf.app.flags.DEFINE_string("data_dir", "data/full_swda_clean_42da_sentiment_dialog_corpus.p", "Raw data directory.")
# # tf.app.flags.DEFINE_string("data_dir", "data/test_data.p", "Raw data directory.") # TODO redirect this to the correct corpus
# tf.app.flags.DEFINE_string("data_dir", "data/dbpedia_small.p", "Raw data directory.") # TODO redirect this to the correct corpus
# tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
# tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
# tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
# tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
# tf.app.flags.DEFINE_string("test_path", "run1500783422", "the dir to load checkpoint for forward only")
# tf.app.flags.DEFINE_string("lda_model_path", "lda/lda_model", "the path to pretrained LDA model")
# tf.app.flags.DEFINE_string("id2word_path", "lda/id2word_wiki.txt", "the path to the id2word dict for LDA model")
# tf.app.flags.DEFINE_string("vocab_dict_path", "data/vocab", "Vocab files directory.")
# tf.app.flags.DEFINE_bool("use_imdb", False, "whether to use the keras imdb dataset")



if __name__ == "__main__":

tf.app.flags.DEFINE_string("work_dir", "working_hier_baseline", "Experiment results directory.")
FLAGS = tf.app.flags.FLAGS

    if FLAGS.forward_only:
        if FLAGS.test_path is None:
            print("Set test_path before forward only")
            exit(1)
    main(HierBaseline, FLAGS)