from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary

import numpy as np

class LDAModel(object):
	def __init__(self, trained_model_path, id2word_path):
		self.model_path = trained_model_path
		self.id2word_path = id2word_path
		self.model = LdaModel.load(self.model_path)
		self.id2word = Dictionary.load_from_text(self.id2word_path)
		self.num_topics = self.model.num_topics

	def tokenize(self, text):
		return([token for token in simple_preprocess(text) if token not in STOPWORDS])

	def get_topic_vector(self, text):
		bow_vector = id2word.doc2bow(tokenize(text))
		lda_result = lda_model[bow_vector]
		topic_vector = np.zeros(self.num_topics)

		for topic_id, val in lda_result:
			topic_vector[topic_id] = val

		return topic_vector