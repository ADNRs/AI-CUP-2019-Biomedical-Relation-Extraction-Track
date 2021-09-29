import re
import keras
import numpy as np
from utils.load import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from keras.layers import Lambda
from keras.callbacks import Callback
from keras_bert import load_vocabulary
from keras_bert import Tokenizer
from keras_bert import get_checkpoint_paths
from keras_bert.datasets import get_pretrained
from keras_bert.datasets import PretrainedList

# preprocess the data
def preprocess(file_path, pred = False, verbose = True):
    if pred:
        PMID, sentence_id, sentence, gene1, gene2, gene1_idx, gene2_idx = \
            load_data(file_path, pred = True, verbose = verbose)
    else:
        PMID, sentence_id, sentence, gene1, gene2, gene1_idx, gene2_idx, re_type = \
            load_data(file_path, pred = False, verbose = verbose)
    sentence, gene1, gene2 = greek2eng(sentence, gene1, gene2, verbose = verbose)
    sentence, gene1, gene2 = upper2lower(sentence, gene1, gene2, verbose = verbose)
    gene1_idx, gene2_idx = correct_index(sentence, gene1, gene2, gene1_idx, gene2_idx, verbose = verbose)
    sentence = rename_target_genes(sentence, gene1, gene2, gene1_idx, gene2_idx, verbose = verbose)
    if pred:
        return sentence
    else:
        return sentence, re_type

# convert Greek alphabets to English alphabets
def greek2eng(sentence = None, gene1 = None, gene2 = None, verbose = True):
    if verbose:
        print('Converting Greek alphabets to English alphabets:')
    greek = {'ß': 'b', 'ζ': 'z', 'λ': 'r', 'Δ': 'D', 'δ': 'd', 'ɛ': 'e', 'σ': 's', 'π': 'p', 'τ': 't'}
    for i in range(len(sentence)):
        for j in greek:
            while j in sentence[i]:
                sentence[i] = sentence[i].replace(j, greek[j])
            while j in gene1[i]:
                gene1[i] = gene1[i].replace(j, greek[j])
            while j in gene2[i]:
                gene2[i] = gene2[i].replace(j, greek[j])
        if verbose and not i % 571:
            print('\r\tProcessed: %.f' % (((i + 1) / len(sentence)) * 100) + '%',  end = '')
    if verbose:
        print('\r\tProcessed: 100%')
    return sentence, gene1, gene2

# convert uppercase letters to lowercase ones
def upper2lower(sentence, gene1, gene2, verbose = True):
    if verbose:
        print('Converting uppercase letters to lowercase ones:')
    for i in range(len(sentence)):
        sentence[i] = sentence[i].lower()
        gene1[i] = gene1[i].lower()
        gene2[i] = gene2[i].lower()
        if verbose and not i % 571:
            print('\r\tProcessed: %.f' % (((i + 1) / len(sentence)) * 100) + '%',  end = '')
    if verbose:
        print('\r\tProcessed: 100%')
    return sentence, gene1, gene2

# correct index errors
def correct_index(sentence, gene1, gene2, gene1_idx, gene2_idx, verbose = True):
    if verbose:
        print('Correcting index errors:')
    for i in range(len(sentence)):
        geneA, temp, geneB, temp2 = *gene1[i].split('|'), *gene2[i].split('|')
        geneA_idx_start, geneA_idx_end = map(int, gene1_idx[i].split('|'))
        geneB_idx_start, geneB_idx_end = map(int, gene2_idx[i].split('|'))
        if sentence[i][geneA_idx_start : geneA_idx_end] != geneA:
            gene1_idx[i] = str(geneA_idx_start + 1) + '|' + str(geneA_idx_end + 1)
        if sentence[i][geneB_idx_start : geneB_idx_end] != geneB:
            gene2_idx[i] = str(geneB_idx_start + 1) + '|' + str(geneB_idx_end + 1)
        if verbose and not i % 571:
            print('\r\tProcessed: %.f' % (((i + 1) / len(sentence)) * 100) + '%',  end = '')
    if verbose:
        print('\r\tProcessed: 100%')
    return gene1_idx, gene2_idx

# rename the two genes to genea and geneb
def rename_target_genes(sentence, gene1, gene2, gene1_idx, gene2_idx, verbose = True):
    if verbose:
        print('Renaming the two genes to genea and geneb:')
    for i in range(len(sentence)):
        geneA, temp, geneB, temp2 = *gene1[i].split('|'), *gene2[i].split('|')
        geneA_slice = slice(*map(int, gene1_idx[i].split('|')))
        geneB_slice = slice(*map(int, gene2_idx[i].split('|')))
        temp = list(sentence[i])
        temp[geneA_slice] = 'A' * len(geneA)
        temp[geneB_slice] = 'B' * len(geneB)
        sentence[i] = ''.join(temp)
        geneA_len = len(gene1[i].split('|')[0])
        geneB_len = len(gene2[i].split('|')[0])
        sentence[i] = sentence[i].replace('A' * geneA_len, 'genea')
        sentence[i] = sentence[i].replace('B' * geneB_len, 'geneb')
        sentence[i] = sentence[i].replace(_get_whole_gene_name(sentence[i], 'genea'), 'genea')
        sentence[i] = sentence[i].replace(_get_whole_gene_name(sentence[i], 'geneb'), 'geneb')
        if verbose and not i % 571:
            print('\r\tProcessed: %.f' % (((i + 1) / len(sentence)) * 100) + '%',  end = '')
    if verbose:
        print('\r\tProcessed: 100%')
    return sentence

def _get_whole_gene_name(sen, gene):
    start = sen.find(gene)
    end = start
    while start > 0 and sen[start - 1].isalpha():
        start -= 1
    while end < len(sen) - 1 and sen[end + 1].isalpha():
        end += 1
    return sen[start:end + 1]

# encode the data into the form BERT accepts
def bert_encode(sentence, verbose = True, maxlen = 100):
    model_path = get_pretrained(PretrainedList.uncased_base)
    paths = get_checkpoint_paths(model_path)
    token_dict = load_vocabulary(paths.vocab)
    token_dict_inv = {v: k for k, v in token_dict.items()}
    tokenizer = Tokenizer(token_dict)
    X_token_ids = []
    X_segment_ids = []
    if verbose:
        print('Tokenizing and encoding:')
    for i in range(len(sentence)):
        token_ids, segment_ids = tokenizer.encode(sentence[i], max_len = maxlen)
        X_token_ids.append(token_ids)
        X_segment_ids.append(segment_ids)
        if verbose and not i % 571:
                print('\r\tProcessed: %.f' % (((i + 1) / len(sentence)) * 100) + '%',  end = '')
    if verbose:
        print('\r\tProcessed: 100%')
    X_token_ids = np.array(X_token_ids)
    X_segment_ids = np.array(X_segment_ids)
    return X_token_ids, X_segment_ids

# encode the re_type
def one_hot_encode(re_type, verbose = True):
    if verbose:
        print('Encoding re_type: ', end = '')
    re_to_id = load_dict('re_to_id')
    for i in range(len(re_type)):
        re_type[i] = re_to_id[re_type[i]]
    re_type = keras.utils.to_categorical(re_type)
    if verbose:
        print('Done')
    return re_type

# decode the encoded re_type
def one_hot_decode(re_type, verbose = True):
    if verbose:
        print('Decoding re_type: ', end = '')
    id_to_re = load_dict('id_to_re')
    re_type = [np.argmax(x) for x in re_type]
    re_type = [id_to_re[x] for x in re_type]
    if verbose:
        print('Done')
    return re_type

# count the number of each re_type
def count_re(re_type, verbose = True):
    if verbose:
        print('Counting relationship: ', end = '')
    re = list(load_dict('re_to_id').keys())
    re_to_num = {x: 0 for x in re}
    for i in re_type:
        re_to_num[i] = re_to_num[i] + 1
    if verbose:
        print('Done')
    return re_to_num

###########################################################################################################
# RemoveMask is not my original work, the related information is given below
#
# Author: Ehud Ben-Reuven
# Availability: https://gist.github.com/udibr/676c742c8843fdcfdfd24f4dcdc3bdfb
###########################################################################################################
class RemoveMask(Lambda):
    def __init__(self):
        super(RemoveMask, self).__init__((lambda x, mask: x))
        self.supports_masking = True
    
    def compute_mask(self, input, input_mask = None):
        return None

###########################################################################################################
# F1_bert is not totally my original work, but I made some changes on it to fit my need
#
# Author: Thong Nguyen
# Availability: https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
###########################################################################################################
class F1_bert(Callback):
    def __init__(self, name):
        self.file_path = './models/' + name + '-weights.h5'
    
    def on_train_begin(self, logs = {}):
        self.val_f1s = []
        self.best_val_f1 = 0
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs = {}):
        boundary = lambda x: [1] if x >= 0.5 else [0]
        val_predict = self.model.predict([self.validation_data[0], self.validation_data[1]])
        val_predict = [np.argmax(x) for x in val_predict]
        val_ans = self.validation_data[2]
        val_ans = [np.argmax(x) for x in val_ans]
        _val_f1 = f1_score(val_ans, val_predict, average = 'macro')
        _val_recall = recall_score(val_ans, val_predict, average = 'macro')
        _val_precision = precision_score(val_ans, val_predict, average = 'macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('val_f1 = %.4f, val_recall = %.4f, val_precision = %.4f' % \
              (_val_f1, _val_recall, _val_precision))
        print('max_f1: %.4f' % max(self.val_f1s))
        if _val_f1 > self.best_val_f1:
            self.model.save_weights(self.file_path, overwrite = True)
            self.best_val_f1 = _val_f1
            print('best_f1: %.4f, saving weights to %s' % (self.best_val_f1, self.file_path))
        else:
            print('val_f1: %.4f, but not the best f1' % _val_f1)
        return