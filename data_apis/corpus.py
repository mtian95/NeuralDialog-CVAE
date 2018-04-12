#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
import pickle as pkl
from collections import Counter
import numpy as np
import nltk
import json
import os
import sys
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import keras 



class SWDADialogCorpus(object):
    dialog_act_id = 0
    sentiment_id = 1
    liwc_id = 2

    def __init__(self, corpus_path, max_vocab_cnt=10000, word2vec=None, word2vec_dim=None, vocab_dict_path=None, 
        lda_model=None, imdb=False):

        """
        :param corpus_path: the folder that contains the SWDA dialog corpus
        """
        self._path = corpus_path
        self.word_vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        self.dialog_id = 0
        self.meta_id = 1
        # self.utt_id = 2
        self.utt_id = 1 
        self.sil_utt = ["<s>", "<sil>", "</s>"]
        self.imdb = imdb
        # self.is_rnn_baseline = is_rnn_baseline
        
        # if not self.imdb:
        #     data = pkl.load(open(self._path, "rb"))
        #     self.train_corpus = self.process(data["train"])
        #     self.valid_corpus = self.process(data["valid"])
        #     self.test_corpus = self.process(data["test"])

        # elif self.imdb:
        #     self.index_from = 3
        #     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        #                                                             path='imdb.npz',
        #                                                             num_words=None,
        #                                                             skip_top=0,
        #                                                             maxlen=None,
        #                                                             seed=113,
        #                                                             start_char=2,
        #                                                             oov_char=1,
        #                                                             index_from=self.index_from)

        #     self.train_corpus = self.process(x_train)
        #     self.valid_corpus = self.process(data["valid"])
        #     self.test_corpus = self.process(data["test"])

        # if self.is_rnn_baseline:
        #     data = pkl.load(open(self._path, "rb"))
        #     self.train_corpus_sent = self.process_baseline(data["train"])
        #     self.valid_corpus_sent = self.process_baseline(data["valid"])
        #     self.test_corpus_sent = self.process_baseline(data["test"])
        #     self.train_corpus = self.process(data["train"])
        #     self.valid_corpus = self.process(data["valid"])
        #     self.test_corpus = self.process(data["test"])

        # else:
        #     data = pkl.load(open(self._path, "rb"))
        #     self.train_corpus = self.process(data["train"])
        #     self.valid_corpus = self.process(data["valid"])
        #     self.test_corpus = self.process(data["test"])

        data = pkl.load(open(self._path, "rb"))
        self.train_corpus = self.process(data["train"])
        self.valid_corpus = self.process(data["valid"])
        self.test_corpus = self.process(data["test"])

        self.vocab_dict_path = vocab_dict_path
        self.lda_model = lda_model
        self.build_vocab(max_vocab_cnt)
        self.load_word2vec()
        print("Done loading corpus")

    def process(self, data):
        """new_dialog: [(a, 1/0), (a,1/0)], new_meta: (a, b, topic), new_utt: [[a,b,c)"""
        """ 1 is own utt and 0 is other's utt"""
        new_dialog = []
        new_meta = []
        new_utts = []
        bod_utt = ["<s>", "<d>", "</s>"] # TODO what do we do about topic for this?
        all_lenes = []

        for l in data:
            lower_utts = [(["<s>"] + nltk.WordPunctTokenizer().tokenize(utt.lower()) + ["</s>"])
                          for utt in l] # for utt in l['utts']]
            all_lenes.extend([len(u) for u in lower_utts])

            # dialog = [(bod_utt, 0)] + [(utt, int(ind==len(lower_utts)-2)) for ind, utt in enumerate(lower_utts)]
            dialog = [(utt, int(ind==len(lower_utts)-2)) for ind, utt in enumerate(lower_utts)]

            new_utts.extend([bod_utt] + lower_utts)
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lenes), float(np.mean(all_lenes))))
        return new_dialog, new_utts

    # def process_baseline(self, data):
    #     """new_dialog: [(a, 1/0), (a,1/0)], new_meta: (a, b, topic), new_utt: [[a,b,c)"""
    #     """ 1 is own utt and 0 is other's utt"""
    #     new_dialog = []
    #     new_meta = []
    #     new_utts = []
    #     bod_utt = ["<s>", "<d>", "</s>"] # TODO what do we do about topic for this?
    #     all_lenes = []

    #     for l in data:
    #         for utt in l:
    #             lower_utts = [(["<s>"] + nltk.WordPunctTokenizer().tokenize(word.lower()) + ["</s>"])
    #                       for word in utt] # for utt in l['utts']]
    #             all_lenes.extend([len(u) for u in lower_utts])

    #             # dialog = [(bod_utt, 0)] + [(utt, int(ind==len(lower_utts)-2)) for ind, utt in enumerate(lower_utts)]
    #             dialog = lower_utts

    #         new_utts.extend([bod_utt] + lower_utts)
    #         new_dialog.append(dialog)

    #     print("Max utt len %d, mean utt len %.2f" % (np.max(all_lenes), float(np.mean(all_lenes))))
    #     return new_dialog, new_utts

    def build_vocab(self, max_vocab_cnt):
        if not self.imdb:
            all_words = []
            for tokens in self.train_corpus[self.utt_id]:
                all_words.extend(tokens)
            vocab_count = Counter(all_words).most_common()
            raw_vocab_size = len(vocab_count)
            discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
            vocab_count = vocab_count[0:max_vocab_cnt]

            # create vocabulary list sorted by count
            print("Load corpus with train size %d, valid size %d, "
                  "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
                  % (len(self.train_corpus), len(self.valid_corpus), len(self.test_corpus),
                     raw_vocab_size, len(vocab_count), vocab_count[-1][1], float(discard_wc) / len(all_words)))

            # make vocab
            self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count] # is a list
            self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)} 
            self.unk_id = self.rev_vocab["<unk>"]

            print("<d> index %d" % self.rev_vocab["<d>"])
            print("<sil> index %d" % self.rev_vocab.get("<sil>", -1))

        elif self.imdb:
            raise NotImplementedError("Need to get imdb to work")
            # word_to_id = imdb.get_word_index()
            # word_to_id = {k:(v + self.index_from) for k,v in word_to_id.items()}
            # word_to_id["<pad>"] = 0
            # word_to_id["<s>"] = 2
            # word_to_id["<unk>"] = 1
            # make word_to_id into a list and that'll be self.vocab. need to leave some blanks in the front
            # self.unk_id = 1
            # self.rev_vocab = {v:k for k,v in word_to_id.items()}

        # NOTE commenting out below lines gets rid of topic vocab and dialog vocab
        # # create topic vocab
        # all_topics = []
        # for a, b, topic in self.train_corpus[self.meta_id]:
        #     all_topics.append(topic)
        # self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
        # self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}
        # print("%d topics in train data" % len(self.topic_vocab))

        # # get dialog act labels
        # all_dialog_acts = []
        # for dialog in self.train_corpus[self.dialog_id]:
        #     all_dialog_acts.extend([feat[self.dialog_act_id] for caller, utt, feat in dialog if feat is not None])

        # all_dialog_acts = []
        # for dialog in self.train_corpus[self.dialog_id]:
        #     all_dialog_acts.extend([feat[self.dialog_act_id] for caller, utt, feat in dialog if feat is not None])
        # self.dialog_act_vocab = [t for t, cnt in Counter(all_dialog_acts).most_common()]
        # self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}
        # print(self.dialog_act_vocab)
        # print("%d dialog acts in train data" % len(self.dialog_act_vocab))

    def load_word2vec(self):
        if self.word_vec_path is None:
            return
        with open(self.word_vec_path, "rb") as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines:
            w, vec = l.split(" ", 1)
            raw_word2vec[w] = vec
        # clean up lines for memory efficiency
        self.word2vec = []
        oov_cnt = 0
        for v in self.vocab:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(self.word2vec_dim) * 0.1
            else:
                vec = np.fromstring(str_vec, sep=" ")
            self.word2vec.append(vec)
        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))

    def get_utt_corpus(self):
        """Currently unused"""
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([self.rev_vocab.get(t, self.unk_id) for t in line])
            return results
        # convert the corpus into ID
        id_train = _to_id_corpus(self.train_corpus[self.utt_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.utt_id])
        id_test = _to_id_corpus(self.test_corpus[self.utt_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}


    def get_dialog_corpus(self):
        """
        Prepare dialog corpus.
        Returns list of lists, one list for each paragraph
        Each list = list of tuples, one for each sentence
        Sentence tuple = ([tokens rep each word in sent], floor, sent topic)
        """
        def _get_topic_vector(text):
            text = " ".join(text)
            return self.lda_model.get_topic_vector(text)

        def _to_id_corpus2(data):
            results = []
            for dialog in data:
                temp_tokens = []
                temp_topics = []
                temp_floor = []
                for utt, end_label in dialog:
                    temp_topics.append(_get_topic_vector(utt))
                    temp_tokens.append([self.rev_vocab.get(t, self.unk_id) for t in utt])
                    temp_floor.append(end_label)
                # temp_floor = [0]*(len(dialog)-1) + [1]
                results.append(zip(temp_tokens, temp_floor, temp_topics))
            return results

        # deprecated method
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                temp = []
                # convert utterance and feature into numeric numbers
                for utt, floor, feat in dialog:
                    if feat is not None:
                        id_feat = list(feat)
                        id_feat[self.dialog_act_id] = self.rev_dialog_act_vocab[feat[self.dialog_act_id]]
                    else:
                        id_feat = None
                    temp.append(([self.rev_vocab.get(t, self.unk_id) for t in utt], floor, id_feat))
                results.append(temp)
            return results

        id_train = _to_id_corpus2(self.train_corpus[self.dialog_id])
        id_valid = _to_id_corpus2(self.valid_corpus[self.dialog_id])
        id_test = _to_id_corpus2(self.test_corpus[self.dialog_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_meta_corpus(self):
        """
        Prepares meta corpus. 
        Returns a paragraph level topic vector for each paragraph (aka each dialog)
        """
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                tmp_sentences = []
                for utt, end_label in dialog:
                    tmp_sentences.extend(utt)
                results.append(self.lda_model.get_topic_vector(" ".join(tmp_sentences)))
            return results

        id_train = _to_id_corpus(self.train_corpus[self.dialog_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.dialog_id])
        id_test = _to_id_corpus(self.test_corpus[self.dialog_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}


    def get_baseline_dialog_corpus(self):
        """
        Prepare corpus for the RNN baseline model. 
        Returns a list of lists, one for each sentence.
        Each sentence list is a list of the tokenized words in that paragraph.
        """
        def _get_topic_vector(text):
            text = " ".join(text)
            return self.lda_model.get_topic_vector(text)

        # def _to_id_corpus(data):
        #     results = []
        #     for sentence in data:
        #         print sentence[0]
        #         temp_tokens = [self.rev_vocab.get(word, self.unk_id) for word in sentence]
        #         results.append(temp_tokens)
        #     return results

        def _to_id_corpus2(data):
            results = []
            for dialog in data:
                for utt, end_label in dialog:
                    temp_tokens = []
                    for word in utt:
                        temp_tokens.append([self.rev_vocab.get(word, self.unk_id)])
                    results.append(temp_tokens)
            return results

        id_train = _to_id_corpus2(self.train_corpus[self.dialog_id])
        id_valid = _to_id_corpus2(self.valid_corpus[self.dialog_id])
        id_test = _to_id_corpus2(self.test_corpus[self.dialog_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}


    def get_baseline_meta_corpus(self):
        """
        Prepares meta corpus for the RNN baseline. 
        Returns a paragraph level topic vector for each sentence. 
        AKA if a paragraph has 5 sentences, that paragraph's topic vector is added 5 times.
        """
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                tmp_sentences = []
                sent_count = 0
                for utt, end_label in dialog:
                    tmp_sentences.extend(utt)
                    sent_count += 1
                results.extend([self.lda_model.get_topic_vector(" ".join(tmp_sentences))]*sent_count)
            return results

        id_train = _to_id_corpus(self.train_corpus[self.dialog_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.dialog_id])
        id_test = _to_id_corpus(self.test_corpus[self.dialog_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}