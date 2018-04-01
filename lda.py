import logging
import itertools
import time
import sys

import numpy as np
import gensim

from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import LdaModel

from config_utils import LDAConfig as Config

def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
            continue  # ignore short articles and various meta-articles
        yield title, tokens

class WikiCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).
        
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs
    
    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return self.clip_docs
        

if __name__ == '__main__':

	logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
	logging.root.level = logging.INFO 

	if Config.corpus_is_wiki:
		# make id2word dictionary
		doc_stream = (tokens for _, tokens in iter_wiki(Config.lda_corpus_path))

		id2word = gensim.corpora.Dictionary(doc_stream)
		print(id2word)
		id2word.filter_extremes(no_below=Config.no_below, no_above=Config.no_above_pct)
		id2word.save_as_text(Config.id2word_path)

		# initialize wiki corpus
		wiki_corpus = WikiCorpus(Config.lda_corpus_path, id2word)

		# store bag of words of the corpus into a file
		gensim.corpora.MmCorpus.serialize(Config.lda_bow_path, wiki_corpus)

		# load mm corpus
		mm_corpus = gensim.corpora.MmCorpus(Config.lda_bow_path)
		print(mm_corpus)

		# training lda model
		if Config.clip_corpus:
			clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, Config.docs_to_clip_to)
			save_file_txt = Config.docs_to_clip_to
		else:
			clipped_corpus = mm_corpus
			save_file_txt = 'all' 
		lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=Config.num_topics, id2word=id2word, passes=4)

		# Save LDA model
		lda_model.save(Config.lda_model_save_path+'_'+save_file_txt+'_'+str(int(time.time())))

	else:
		raise NotImplementedError("LDA model can't handle non-wiki corpus yet")