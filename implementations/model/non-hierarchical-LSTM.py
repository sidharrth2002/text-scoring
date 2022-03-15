'''
Tensorflow implementation of non-hierarchical LSTM
'''
import os
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.utils import to_categorical
from implementations.model.layers import Attention, ZeroMaskedEntries

working_directory = '.'

A_train = pd.read_csv('./train.csv', index_col=0)
A_test = pd.read_csv('./test.csv', index_col=0)
A_train.head()

glove_file = open(os.path.join(working_directory, 'glove.6B.300d.txt'))
glove_embeddings = {}
for line in glove_file:
    temp = line.split(" ")
    glove_embeddings[temp[0]] = np.asarray([float(i) for i in temp[1:]])
print("Loaded GLoVE")

EMBEDDING_DIM = 300
MAX_WORDS = 6000
MAX_SEQUENCE_LENGTH = 700
VALIDATION_SPLIT = 0.2
DELTA = 20

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(pd.concat([A_train, A_test], axis=0)['explanation_practice'].to_list())
sequences = tokenizer.texts_to_sequences(A_train['explanation_practice'].to_list())
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(sequences.shape)

embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))
print(embedding_matrix.shape)

for word,i in word_index.items():
    if(i>=len(word_index)):
        continue
    if word in glove_embeddings:
        embedding_matrix[i]=glove_embeddings[word]

vocab_size = len(word_index)
print(vocab_size)

from tensorflow.keras.layers import Input, Bidirectional, Embedding, LSTM, Dense, Flatten, Dropout, concatenate, GlobalAveragePooling1D, Conv1D, TimeDistributed, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

def get_model():
    input_layer = Input(name='response', shape=(MAX_SEQUENCE_LENGTH,))

    embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], mask_zero=True, trainable=True)
    word_embeddings = embedding_layer(input_layer)
    word_embeddings_masked = ZeroMaskedEntries(name='pos_x_maskedout')(word_embeddings)

    first_convolution = Conv1D(50, 3, padding='valid')(word_embeddings_masked)

    first_lstm_layer = LSTM(300, return_sequences=True, recurrent_dropout=0.4, dropout=0.4)(first_convolution)
    first_dropout = Dropout(0.5)(first_lstm_layer)

    lstm_means = Attention()(first_dropout)
    embedding_dense = Dense(64)(lstm_means)

    applied_input = Input(name='applied', shape=(1,))
    applied_dense = Dense(4)(applied_input)

    concatenated = Concatenate()([embedding_dense, applied_dense])
    second_dropout = Dropout(0.3)(concatenated)
    score_dense = Dense(32)(second_dropout)
    score_final = Dense(5, activation='softmax', name='score')(score_dense)

    inputs = {'response': input_layer, 'whether_criteria_applied': applied_input}
    outputs = {'score': score_final}

    model = Model(inputs=inputs, outputs=outputs, name='Taghipour')
    model.emb_index = 0

    loss = {'score': CategoricalCrossentropy()}
    metric = {'score': CategoricalAccuracy('accuracy')}

    optimizer = Adam()

    model.compile(loss=loss, optimizer=optimizer, metrics=metric)

    model.summary()

    return model

model = get_model()

history = model.fit(x={'response': sequences, 'whether_criteria_applied': A_train['applied']}, y={'score': to_categorical(A_train['user_score'])}, batch_size=64, epochs=30, callbacks=[model_checkpoint_callback], validation_data=({'response': test_sequences, 'whether_criteria_applied': A_test['applied']}, {'score': to_categorical(A_test['user_score'])}))

test_sequences = tokenizer.texts_to_sequences(A_test['explanation_practice'].to_list())

pred = model.predict(x={'response': test_sequences, 'whether_criteria_applied': A_test['applied']})['score'].argmax(axis=1)

print(cohen_kappa_score(pred, A_test['user_score'].to_list(), weights='quadratic'))