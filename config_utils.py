#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University


class KgCVAEConfig(object):
    description= None
    use_hcf = True  # use dialog act in training (if turn off kgCVAE -> CVAE)
    update_limit = 3000  # the number of mini-batch before evaluating the model

    # how to encode utterance.
    # bow: add word embedding together
    # rnn: RNN utterance encoder
    # bi_rnn: bi_directional RNN utterance encoder
    sent_type = "bi_rnn"

    # latent variable (gaussian variable)
    latent_size = 200  # the dimension of latent variable
    full_kl_step = 10000  # how many batch before KL cost weight reaches 1.0
    dec_keep_prob = 1.0  # do we use word drop decoder [Bowman el al 2015]

    # Network general
    cell_type = "gru"  # gru or lstm
    embed_size = 200  # word embedding size
    topic_embed_size = 30  # topic embedding size
    da_embed_size = 30  # dialog act embedding size
    cxt_cell_size = 600  # context encoder hidden size
    sent_cell_size = 300  # utterance encoder hidden size
    dec_cell_size = 400  # response decoder hidden size
    backward_size = 10  # how many utterance kept in the context window
    step_size = 1  # internal usage
    max_utt_len = 25  # max number of words in an utterance
    num_layer = 1  # number of context RNN layers

    # Optimization parameters
    op = "adam"
    grad_clip = 5.0  # gradient abs max cut
    init_w = 0.08  # uniform random from [-init_w, init_w]
    batch_size = 30  # mini-batch size
    init_lr = 0.001  # initial learning rate
    lr_hold = 1  # only used by SGD
    lr_decay = 0.6  # only used by SGD
    keep_prob = 1.0  # drop out rate
    improve_threshold = 0.996  # for early stopping
    patient_increase = 2.0  # for early stopping
    early_stop = True
    max_epoch = 60  # max number of epoch of training # TODO change this back to not 1 (60)

    grad_noise = 0.0  # inject gradient noise?

    # Topic model related
    num_topics = 100

    # RNN baseline related
    max_rnn_sent_len = 25
    rnn_backward_size = max_rnn_sent_len


class LDAConfig(object):
    corpus_is_wiki = True # whether LDA model corpus is a wikipedia or simple-wiki corpus
    lda_corpus_path = './data/simplewiki-20171020-pages-articles-multistream.xml.bz2' # corpus for LDA model. End in .bz2
    clip_corpus = False # clip corpus to smaller corpus? Will save training time
    id2word_path = 'lda2/id2word_wiki.txt' # path for id2word dictionary
    lda_bow_path = 'lda2/wiki_bow.mm' # path of bow of corpus

    no_below = 20 # filter words that appear in less than this many documents
    no_above_pct = 0.1 # filter words that appear in more than this percent of documents
    docs_to_clip_to = 10 # number of documents to train LDA model on
    num_topics = 10

    lda_model_save_path = 'lda2/lda_model' # beginning of path that trained lda model will be saved to







