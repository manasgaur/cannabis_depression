import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

import fastText as fasttext
import csv

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs

MAX_NB_WORDS = 100000   #randomly chosen--used for word_seq generator
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

df = pd.read_csv('normalized_tweets.csv')
df = df[df.label != 8]   #dropping label '8' tweets

df['doc_len'] = df['tweet'].apply(lambda words: len(words.split(" ")))    #length of each tweet

max_seq_len = np.round(df['doc_len'].mean() + df['doc_len'].std()).astype(int)

raw_docs = df['tweet'].tolist()

#processed docs generator
processed_docs = []
for doc in tqdm(raw_docs):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs.append(" ".join(filtered))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_docs)  
word_seq = tokenizer.texts_to_sequences(processed_docs)   #each tweet gets tokenized---word_seq is a list of tokenized tweets(length = 3059)

word_seq = sequence.pad_sequences(word_seq, maxlen=max_seq_len)

word_index = tokenizer.word_index   #dictionary of words in tweets and their associated id
inverted_word_index = dict((v, k) for k, v in word_index.iteritems())   #used for deriving word from an index(used to deal with padded sequences)

embed_dim = 300 

model_bin="cc.en.300.bin"   #fasttext model
model=fasttext.load_model(model_bin) #loading fasttext model


#function to get predecessor and successor tokens given a current token
def find_prev_next(elem, elements):
    previous, next = None, None
    index = elements.index(elem)
    if index > 0:
        previous = elements[index -1]
    if index < (len(elements)-1):
        next = elements[index +1]
    return previous, next

#word embedding generator
def word_embedding(idx, seq):
	seq=seq.tolist()
	try:
		word = inverted_word_index[idx]
		current_embedding=model.get_word_vector(word)
	except:
		current_embedding=np.zeros(300,)
	previous,next = find_prev_next(idx,seq)
	try:
		word_previous = inverted_word_index[previous]
		prev_embedding=model.get_word_vector(word_previous)
	except:
		prev_embedding=np.zeros(300,)
	try:
		word_next = inverted_word_index[next]
		next_embedding=model.get_word_vector(word_next)
	except:
		next_embedding=np.zeros(300,)
	conc_WF=np.concatenate((prev_embedding,current_embedding,next_embedding), axis=0)
	return conc_WF    #900-d word embedding---can be reduced to 300-d via tSVD or PCA

#position embedding generator
def position_embedding(idx,seq):
	seq=seq.tolist()
	try:
		cannabis_idx = seq.index(1)
	except:
		cannabis_idx = 0
	token_idx = seq.index(idx)
	try:
		depression_idx=seq.index(2)
	except:
		depression_idx = 0
	token_cannabis = token_idx - cannabis_idx
	token_depression = token_idx - depression_idx
	if token_cannabis > 0:
		tc_rand = np.random.randint(0, token_cannabis, size=(100,))
	elif token_cannabis < 0:
		tc_rand=np.random.randint(token_cannabis, 0, size=(100,))
	elif token_cannabis == 0:
		token_cannabis = 1
		tc_rand=np.random.randint(0, token_cannabis, size=(100,))
	if token_depression > 0:
		td_rand=np.random.randint(0, token_depression, size=(100,))
	elif token_depression < 0:
		td_rand=np.random.randint(token_depression, 0, size=(100,))
	elif token_depression == 0:
		token_depression = 1
		td_rand=np.random.randint(0, token_depression, size=(100,))
	conc_PF=np.concatenate((tc_rand,td_rand))
	return conc_PF   #200-d positional feature embedding

word_embedding_matrix = np.random.random((len(word_seq) * len(word_seq[0]), 900))
position_embedding_matrix = np.random.random((len(word_seq) * len(word_seq[0]), 200))


for seq in word_seq:
	for idx in seq:
		word_embedding_matrix[idx] = word_embedding(idx, seq)

for seq in word_seq:
	for idx in seq:
		position_embedding_matrix[idx] = position_embedding(idx, seq)

word_output_dim = 300
word_embedding_layer = Embedding(len(word_seq) * len(word_seq[0]), word_output_dim, 
	weights = [word_embedding_matrix], input_length = max_seq_len, trainable=False)

position_output_dim = 100
position_embedding_layer = Embedding(len(word_seq) * len(word_seq[0]), position_output_dim,
	input_length = max_seq_len, trainable=True)


final_embedding = concatenate([word_embedding_layer, position_embedding_layer], axis = 1)

#CNN models follows

#training params
batch_size = 256 
num_epochs = 8 

#model parameters
num_filters = 64 
embed_dim = 300 
weight_decay = 1e-4




