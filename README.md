# Usage V2

All model parameters can be changed in config_utils.py

## Train LDA topic model
    python lda.py
Specify the number of topics, number of documents to train on, corpus, and save paths in config_utils.py.  
This will save a trained lda model to lda_model_save_path + number of docs it was trained on + epoch time of runtime

## Train Hierarchical baseline
    python main.py --model_type=hierbaseline --data_dir=[path to pickle file with data format specified below]
All config for this model is the same as the config for KgCVAE set in config_utils.py.  
This will default to training and saving the model into ./working_hierbaseline

## Train CVAE
    python main.py --model_type=cvae --data_dir=[path to pickle file with data format specified below]
will run default cvae training and save model to ./working_cvae

## Train KgCVAE
    python main.py --model_type=kgcvae --data_dir=[path to pickle file with data format specified below]
will run default training and save model to ./working_kgcvae


# ---- Old readme starts here ----


# Knowledge-Guided CVAE for dialog generation

We provide a TensorFlow implementation of the CVAE-based dialog model described in
**Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders**, accepted as a long paper in ACL 2017.
See the [paper](https://arxiv.org/abs/1703.10960) for more details.

# Prerequisites
 - TensorFlow 1.3.0
 - cuDNN 6
 - Python 2.7
 - Numpy
 - NLTK
 - You may need to pip install beeprint if the module is missing

# Usage
## Train a new model
    python main.py --model_type=kgcvae
will run default training and save model to ./working

## Test a existing model
Modify the TF flags at the top of main.py as follows to run a existing model

    forward_only: False -> True
    test_path: set to the folder contains the model. E.g. runxxxx
Then you can run the model by:

    python main.py --model_type=kgcvae
The outputs will be printed to stdout and generated responses will be saved at test.txt in the test_path.

## Use pre-trained Word2vec
Download Glove word embeddings from https://nlp.stanford.edu/projects/glove/
The default setting use 200 dimension word embedding trained on Twitter.

At last, set **word2vec_path** at line 15 of main.py.

## Dataset
We release two dataset:

1. full_swda_clean_42da_sentiment_dialog_corpus.p is a binary dump using python Pickle library that contains the raw data and used for training
2. json_format: the same dialog data also is presented in JSONL format in the data directory.
3. test_mutl_ref.json is only the test data set with multiple references responses with dialog act annotations. The multiple referneces are collected
according to the method described in the Appendix of the paper.

# Data Format
If you want to train the model on your own data. Please create a pickle file has the following format:

    # The top directory is a python dictionary
    type(data) = dict
    data.keys() = ['train', 'valid', 'test']

    # Train/valid/test is a list, each element is one dialog
    train = data['train']
    type(train) = list

    # Each dialog is a dict
    dialog = train[0]
    type(dialog)= dict
    dialog.keys() = ['A', 'B', 'topic', 'utts']

    # A, B contain meta info about speaker A and B.
    # topic defines the dialog prompt topic in Switchboard Corpus.

    # utts is a list, each element is a tuple that contain info about an utterance
    utts = dialog['utts']
    type(utts) = list
    utts[0] = ("A" or "B", "utterance in string", [dialog_act, other_meta_info])

    # For example, a utterance look like this:
    ('B','especially your foreign cars',['statement-non-opinion'])

Put the resulting file into ./data and set the **data_dir** in main.py


# References 
If you use any source codes or datasets included in this toolkit in your
work, please cite the following paper. The bibtex are listed below:
 
    [Zhao et al, 2017]:
     @article{zhao2017learning,
      title={Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders},
      author={Zhao, Tiancheng and Zhao, Ran and Eskenazi, Maxine},
      journal={arXiv preprint arXiv:1703.10960},
      year={2017}
    }
