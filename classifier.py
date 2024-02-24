import json
import re
import sys
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from attention import AttentionWithContext
from data_gen import Corpus

# Modify this paths as well
DATA_DIR = '/home/avu/Pycharm/news-analyzer/models/blstm/data/'
TRAIN_FILE = 'train_set.csv'
TRAIN_LABS = 'train_set_labels_small.csv'
EMBEDDING_FILE = '/home/avu/Pycharm/Document-Classifier-LSTM/glove.6B.200d.txt'
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 100000
# Max number of words in each abstract.
MAX_SEQUENCE_LENGTH = 1000  # MAYBE BIGGER
# This is fixed.
EMBEDDING_DIM = 200
# The name of the model.
STAMP = 'doc_blstm'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def f1_score(y_true, y_pred):
    """
	Compute the micro f(b) score with b=1.
	"""
    y_true = tf.cast(y_true, "float32")
    y_pred = tf.cast(tf.round(y_pred), "float32")  # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred

    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)

    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 2 * precision * recall / (precision + recall)
    f_score = tf.where(tf.math.is_nan(f_score), tf.zeros_like(f_score), f_score)

    return tf.reduce_mean(f_score)

def load_data_test(set):
    class_file = open('class_dict.json', 'r')
    class_json = json.load(class_file)

    X_data = []
    y_data = []

    counter = 0
    for c, (vector, target) in enumerate(set):
        if target[0] in class_json:
            X_data.append(vector)
            y_data.append(target)
            counter += 1

    file = open('tokenizer.json', 'r')
    config = json.loads(file.read())
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(config)

    X_data = tokenizer.texts_to_sequences(X_data)

    X_data = pad_sequences(X_data,
                           maxlen=MAX_SEQUENCE_LENGTH,
                           padding='post',
                           truncating='post',
                           dtype='float32')



    y_data_int = []
    for y_seq in y_data:
        for y in y_seq:
            class_label = class_json[y]
            y_data_int.append([class_label])

    mlb = MultiLabelBinarizer()
    mlb.fit([list(class_json.values())])
    y_data = mlb.transform(y_data_int)

    return X_data, y_data


def load_data(train_set):
    """
	"""

    X_data = []
    y_data = []
    for c, (vector, target) in enumerate(train_set):
        if c % 8 == 0:
            X_data.append(vector)
            y_data.append(target)
            if c % 10000 == 0:
                print(c)


    print((len(X_data), 'training examples'))

    class_freqs = Counter([y for y_seq in y_data for y in y_seq]).most_common()

    class_list = [y[0] for y in class_freqs]
    nb_classes = len(class_list)
    print((nb_classes, 'classes'))
    class_dict = dict(zip(class_list, np.arange(len(class_list))))

    with open('class_dict.json', 'w') as fp:
        json.dump(class_dict, fp, cls=NpEncoder)
    print('Exported class dictionary')

    y_data_int = []
    for y_seq in y_data:
        y_data_int.append([class_dict[y] for y in y_seq])

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                          oov_token=1)
    tokenizer.fit_on_texts(X_data)
    X_data = tokenizer.texts_to_sequences(X_data)

    X_data = pad_sequences(X_data,
                           maxlen=MAX_SEQUENCE_LENGTH,
                           padding='post',
                           truncating='post',
                           dtype='float32')
    print(('Shape of data tensor:', X_data.shape))

    word_index = tokenizer.word_index
    print(('Found %s unique tokens' % len(word_index)))
    with open('word_index.json', 'w') as fp:
        json.dump(word_index, fp)
    print('Exported word dictionary')

    with open('tokenizer.json', 'w') as file:
        tokenizer_json = tokenizer.to_json()
        json.dump(tokenizer_json, file)

    mlb = MultiLabelBinarizer()
    mlb.fit([list(class_dict.values())])
    y_data = mlb.transform(y_data_int)

    print(('Shape of label tensor:', y_data.shape))

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data,
                                                      train_size=0.8,
                                                      test_size=0.2,
                                                      random_state=42)

    return X_train, X_val, y_train, y_val, nb_classes, word_index


def prepare_embeddings(wrd2id):
    """
	"""

    vocab_size = MAX_NB_WORDS
    print(("Found %s words in the vocabulary." % vocab_size))

    embedding_idx = {}
    glove_f = open(EMBEDDING_FILE)
    for line in glove_f:
        values = line.split()
        wrd = values[0]
        coefs = np.asarray(values[1:],
                           dtype='float32')
        embedding_idx[wrd] = coefs
    glove_f.close()
    print(("Found %s word vectors." % len(embedding_idx)))

    embedding_mat = np.random.rand(vocab_size + 1, EMBEDDING_DIM)

    wrds_with_embeddings = 0
    # Keep the MAX_NB_WORDS most frequent tokens.
    for wrd, i in wrd2id.items():
        if i > vocab_size:
            continue

        embedding_vec = embedding_idx.get(wrd)
        # words without embeddings will be left with random values.
        if embedding_vec is not None:
            wrds_with_embeddings += 1
            embedding_mat[i] = embedding_vec

    print((embedding_mat.shape))
    print(('Words with embeddings:', wrds_with_embeddings))

    return embedding_mat, vocab_size


def build_model(nb_classes,
                word_index,
                embedding_dim,
                seq_length,
                stamp):
    """
	"""

    embedding_matrix, nb_words = prepare_embeddings(word_index)

    input_layer = Input(shape=(seq_length,),
                        dtype='int32')

    embedding_layer = Embedding(input_dim=nb_words + 1,
                                output_dim=embedding_dim,
                                input_length=seq_length,
                                weights=[embedding_matrix],
                                embeddings_regularizer=regularizers.l2(0.00),
                                trainable=True)(input_layer)

    drop1 = SpatialDropout1D(0.3)(embedding_layer)

    lstm_1 = Bidirectional(LSTM(128, name='blstm_1',
                                activation='tanh',
                                recurrent_activation='sigmoid',
                                recurrent_dropout=0.0,
                                dropout=0.5,
                                kernel_initializer='glorot_uniform',
                                return_sequences=True),
                           merge_mode='concat')(drop1)
    lstm_1 = BatchNormalization()(lstm_1)

    att_layer = AttentionWithContext()(lstm_1)

    drop3 = Dropout(0.5)(att_layer)

    predictions = Dense(nb_classes, activation='sigmoid')(drop3)

    model = Model(inputs=input_layer, outputs=predictions)

    adam = Adam(learning_rate=0.001)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score])

    model.summary()
    print(stamp)

    # Save the model.
    model_json = model.to_json()
    with open(stamp + ".json", "w") as json_file:
        json_file.write(model_json)

    return model


def load_model(stamp):
    """
	"""

    json_file = open(stamp + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, {'AttentionWithContext': AttentionWithContext})

    model.load_weights(stamp + '.h5')
    print("Loaded model from disk")

    model.summary()

    adam = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score])

    return model


if __name__ == '__main__':

    load_previous = sys.argv[1]

    print(load_previous)

    if load_previous == 'load':
        load_previous = True
    else:
        load_previous = False

    train_set = Corpus(DATA_DIR + TRAIN_FILE, DATA_DIR + TRAIN_LABS)

    X_train, X_val, y_train, y_val, nb_classes, word_index = load_data(train_set)

    if load_previous:
        model = load_model(STAMP)
    else:
        model = build_model(nb_classes,
                            word_index,
                            EMBEDDING_DIM,
                            MAX_SEQUENCE_LENGTH,
                            STAMP)

    monitor_metric = 'f1_score'

    early_stopping = EarlyStopping(monitor=monitor_metric,
                                   patience=5,
                                   verbose=1,
                                   mode='max',
                                   start_from_epoch=5)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path,
                                       monitor=monitor_metric,
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max',
                                       save_weights_only=True)

    hist = model.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=100000,
                     batch_size=8,
                     shuffle=True,
                     callbacks=[model_checkpoint, early_stopping])

    print((hist.history))
