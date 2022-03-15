import os
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Bidirectional, Embedding, LSTM, Dense, Flatten, Dropout, concatenate, GlobalAveragePooling1D, Conv1D, TimeDistributed, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from implementations.model.layers import AttentionLayer, AttentionWithContext

working_directory = '.'
data = pd.read_csv('./data.csv')

max_senten_len = 100
EMBEDDING_DIM = 300
max_senten_num = 50

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
word_index = tokenizer.word_index

glove_file = open(os.path.join(working_directory, 'glove.6B.300d.txt'))
glove_embeddings = {}
for line in glove_file:
    temp = line.split(" ")
    glove_embeddings[temp[0]] = np.asarray([float(i) for i in temp[1:]])
print("Loaded GLoVE")

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

count = 0
for word,i in word_index.items():
    if(i>=len(word_index)):
        continue
    if word in glove_embeddings:
      count += 1
      embedding_matrix[i]=glove_embeddings[word]

vocab_size = len(word_index)

embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM,weights=[embedding_matrix], input_length=max_senten_len, trainable=False)

def get_lstm_lstm():
  applied_input = Input(name='applied', shape=(1,))
  applied_dense = Dense(16)(applied_input)

  word_input = Input(shape=(max_senten_len,), dtype='float32')
  word_sequences = embedding_layer(word_input)
  word_lstm = LSTM(100, return_sequences=True)(word_sequences)
  word_att, word_coeffs = AttentionLayer(EMBEDDING_DIM,return_coefficients=True)(word_lstm)
  wordEncoder = Model(word_input, word_att)

  sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
  sent_encoder = TimeDistributed(wordEncoder)(sent_input)
  sent_lstm = LSTM(100, return_sequences=True)(sent_encoder)
  sent_att, sent_coeffs = AttentionLayer(EMBEDDING_DIM,return_coefficients=True)(sent_lstm)
  sent_dropout = Dropout(0.5)(sent_att)

  concatenated = Concatenate()([sent_dropout, applied_input])
  preds = Dense(5, activation='softmax')(concatenated)


  inputs = {'response': sent_input, 'whether_criteria_applied': applied_input}

  model = Model(inputs, preds)
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
  return model

def get_bilstm_bilstm():
  applied_input = Input(name='applied', shape=(1,))
  applied_dense = Dense(16)(applied_input)

  word_input = Input(shape=(max_senten_len,), dtype='float32')
  word_sequences = embedding_layer(word_input)
  word_lstm = Bidirectional(LSTM(100, return_sequences=True))(word_sequences)
  word_att, word_coeffs = AttentionLayer(EMBEDDING_DIM,return_coefficients=True)(word_lstm)
  wordEncoder = Model(word_input, word_att)

  sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
  sent_encoder = TimeDistributed(wordEncoder)(sent_input)
  sent_lstm = Bidirectional(LSTM(100, return_sequences=True))(sent_encoder)
  sent_att, sent_coeffs = AttentionLayer(EMBEDDING_DIM,return_coefficients=True)(sent_lstm)
  sent_dropout = Dropout(0.5)(sent_att)

  concatenated = Concatenate()([sent_dropout, applied_input])
  preds = Dense(5, activation='softmax')(concatenated)

  inputs = {'response': sent_input, 'whether_criteria_applied': applied_input}

  model = Model(inputs, preds)
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc', 'Precision'])
  return model

def get_cnn_lstm():
  word_input = Input(shape=(max_senten_len,), dtype='float32')
  word_sequences = embedding_layer(word_input)
  word_conv = Conv1D(100, 5, padding='valid')(word_sequences)
  # word_lstm = Bidirectional(LSTM(150, return_sequences=True))(word_sequences)
  # word_dense = TimeDistributed(Dense(200))(word_lstm)
  word_att = AttentionWithContext()(word_conv)
  wordEncoder = Model(word_input, word_att)

  sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
  sent_encoder = TimeDistributed(wordEncoder)(sent_input)
  sent_lstm = Bidirectional(LSTM(150, return_sequences=True))(sent_encoder)
  sent_dense = TimeDistributed(Dense(200))(sent_lstm)
  sent_att = Dropout(0.5)(AttentionWithContext()(sent_dense))
  preds = Dense(5, activation='softmax')(sent_att)
  model = Model(sent_input, preds)
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
  return model
