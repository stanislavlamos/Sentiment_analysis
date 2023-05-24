from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM, SpatialDropout1D
from keras.models import Model
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf

import utils

glove_vectors_path = "../pretrained_vectors/glove.6B.100d.txt"
embedding_dim = 100
tokenizer = Tokenizer(num_words=6000)
maxlen = 200


def tokenize_sentences(X):
    tokenizer.fit_on_texts(X)
    vocab_size = 10000

    X = tokenizer.texts_to_sequences(X)
    new_X = pad_sequences(X, padding='post', maxlen=maxlen)

    return new_X, vocab_size


def process_embeddings(vocab_size):
    glove_file = open(glove_vectors_path, encoding="utf8")
    embeddings_dictionary = dict()

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return embedding_matrix, embeddings_dictionary


def train_lstm_multilabel(vocab_size, labels_count, X_train, y_train, X_val, y_val, lr=0.001, epochs=25):
    embedding_matrix, embeddings_dictionary = process_embeddings(vocab_size)

    deep_inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)(deep_inputs)
    LSTM_Layer_1 = LSTM(128)(embedding_layer)
    dense_layer_1 = Dense(labels_count, activation='sigmoid')(LSTM_Layer_1)
    model = Model(inputs=deep_inputs, outputs=dense_layer_1)

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['acc'])
    model.fit(np.array(X_train), np.array(y_train), batch_size=32, epochs=epochs, verbose=1, validation_data=(np.array(X_val), np.array(y_val)))

    return model


def train_lstm_onelabel(vocab_size, labels_count, X_train, y_train, X_val, y_val, lr=0.001, epochs=25):
    embedding_matrix, embeddings_dictionary = process_embeddings(vocab_size)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(labels_count, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['acc'])
    model.fit(np.array(X_train), np.array(y_train), batch_size=32, epochs=epochs, verbose=1, validation_data=(np.array(X_val), np.array(y_val)))

    return model


def evaluate_lstm(model, X_test, y_test, threshold, mapping_dict):
    y_pred = model.predict(X_test)
    y_pred_onehot = utils.probs_over_threshold(y_pred, threshold)
    print(len(y_pred), len(y_pred[0]))
    print(y_pred_onehot)
    print(len(y_pred_onehot), len(y_pred_onehot[0]))
    score = model.evaluate(np.array(X_test), np.array(y_test))
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])
    print(classification_report(y_test, y_pred_onehot, target_names=mapping_dict.keys(), zero_division=0))
