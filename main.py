#!/usr/bin/env python
# coding: utf-8
import keras
import os
import pandas as pd
from utils.process import *
from utils.load import *
from sklearn.model_selection import train_test_split
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import get_checkpoint_paths
from keras_bert.datasets import get_pretrained
from keras_bert.datasets import PretrainedList
from keras import Input
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers import Convolution1D
from keras.layers import MaxPool1D
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.filterwarnings('ignore')

# maxlen indicating the max length of the sentence length (counting by words not characters)
maxlen = 100
# if the training has already been done, skip the preprocessing of the data for training
if not os.path.isfile('./models/bert-bilstm512-textcnn384.3.4.5-100-weights.h5'):
    # preprocess the train data, and modify the data into the form that the model accepts
    sentence, re_type = preprocess('training.tsv')
    X_token, X_segment, y = *bert_encode(sentence, maxlen = maxlen), one_hot_encode(re_type)
    # split the data into training set and validation set
    X_train_token, X_val_token, X_train_segment, X_val_segment, y_train, y_val = \
        train_test_split(X_token, X_segment, y, test_size = 0.1, random_state = 8787)

# get the path of pretrained weights of BERT (using uncased base, if you don't substitute "pretained.py"
# for the file with the same name in <your keras-bert installation directory>, you will get an
# error message here)
model_path = get_pretrained(PretrainedList.uncased_base)
paths = get_checkpoint_paths(model_path)

# load pretrained BERT
Bert = load_trained_model_from_checkpoint(
    paths.config,
    paths.checkpoint,
    seq_len = maxlen,
    training = False,
    use_adapter = True,
    trainable = \
        ['Encoder-{}-MultiHeadSelfAttention-Adapter'.format(i + 1) for i in range(12)] + \
        ['Encoder-{}-FeedForward-Adapter'.format(i + 1) for i in range(12)] + \
        ['Encoder-{}-MultiHeadSelfAttention-Norm'.format(i + 1) for i in range(12)] + \
        ['Encoder-{}-FeedForward-Norm'.format(i + 1) for i in range(12)]
    )
Bert.name = 'bert_embedding'
Bert.summary()

# build model
# input layer
token_inputs = Input(shape = (maxlen, ))
segment_inputs = Input(shape = (maxlen, ))
# BERT embedding
x = Bert([token_inputs, segment_inputs])
# mask removing (because CuDNNLSTM does not support masking)
x = RemoveMask()(x)
# bi-LSTM
x = Bidirectional(CuDNNLSTM(512, return_sequences = True))(x)
# textCNN
x1 = Convolution1D(384, 3, padding = 'same', strides = 1, activation = 'relu')(x)
x1 = MaxPool1D(pool_size = 4)(x1)
x2 = Convolution1D(384, 4, padding = 'same', strides = 1, activation = 'relu')(x)
x2 = MaxPool1D(pool_size = 4)(x2)
x3 = Convolution1D(384, 5, padding = 'same', strides = 1, activation = 'relu')(x)
x3 = MaxPool1D(pool_size = 4)(x3)
x = Concatenate(axis = -1)([x1, x2, x3])
x = Flatten()(x)
x = Dropout(0.25)(x)
# output layer
outputs = Dense(len(load_dict('re_to_id')), activation = 'softmax')(x)
# create the model
model = Model([token_inputs, segment_inputs], outputs)
model.compile(loss = 'categorical_crossentropy',
              optimizer = Nadam(1e-4),
              metrics = ['accuracy']
             )
model.summary()

# if the training has already been done, skip training
if not os.path.isfile('./models/bert-bilstm512-textcnn384.3.4.5-100-weights.h5'):
    # this can show us the F1-measure on validation set on the end of each epoch, and save the weights that
    # has the highest val_f1
    f1 = F1_bert('bert-bilstm512-textcnn384.3.4.5-100')
    # reduce learning rate when val_loss did not decreased
    reducelr = ReduceLROnPlateau(monitor = 'val_loss',
                                 factor = 0.1,
                                 patience = 1,
                                 verbose = 1,
                                )
    # stop training when val_loss did not decrease for three epochs
    earlystop = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1)
    callbacks_list = [reducelr, earlystop, f1]
    # train the model
    model.fit([X_train_token, X_train_segment], y_train,
              epochs = 50,
              batch_size = 32,
              validation_data = ([X_val_token, X_val_segment], y_val),
              shuffle = True,
              callbacks = callbacks_list
             )

# load the model with the best weights
model_name = 'bert-bilstm512-textcnn384.3.4.5-100-weights.h5'
model.load_weights('./models/' + model_name)

# make prediction
# preprocess the test data
sentence = preprocess('test.tsv', pred = True)
X_token, X_segment = bert_encode(sentence, maxlen = maxlen)
print('Predicting (it would take minutes to finish):')
y_pred = model.predict([X_token, X_segment], verbose = 1)
y_pred = one_hot_decode(y_pred)
# save the prediction into "prediction.tsv"
test = pd.read_csv('./datasets/test.tsv', sep = '\t')
test.insert(7, 'RE_Type', y_pred)
test.to_csv('prediction.tsv', sep = '\t', index = False)

# show you the prediction the model made
pred_count = count_re(y_pred, verbose = 0)
print('-' * 55)
print('Prediction of Test Data')
print('%-30s %-7s' % ('Relationship', 'Prediction'))
print('-' * 55)
for i in pred_count.items():
    print('%-30s  %-6s' % (i[0], i[1]))
