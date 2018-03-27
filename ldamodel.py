from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess

import numpy as np

class LDAModel(object):
	def __init__(self, config, trained_model_path, id2word_path):
		self.model_path = trained_model_path
		self.id2word_path = id2word_path
		self.model = LdaModel.load(self.model_path)
		self.id2word = Dictionary.load_from_text(self.id2word_path)
		self.num_topics = config.num_topics
		assert self.model.num_topics == self.num_topics
	@staticmethod
	def tokenize(text):
		return([token for token in simple_preprocess(text) if token not in STOPWORDS])

	def get_topic_vector(self, text):
		bow_vector = self.id2word.doc2bow(self.tokenize(text))
		lda_result = self.model[bow_vector]
		topic_vector = np.zeros(self.num_topics)

		for topic_id, val in lda_result:
			topic_vector[topic_id] = val
		return topic_vector


	# TODO make a method that can tell you what words constitute a topic vector