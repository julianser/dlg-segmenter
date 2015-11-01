"""
Takes as input a model prototype (linked to a particular dataset) and computes relevant dataset statistics.

Example use:

    THEANO_FLAGS=mode=FAST_COMPILE,floatX=float32 python compute_dataset_statistics.py prototype_movies

@author Iulian Vlad Serban
"""

from data_iterator import *
from state import *
from dialog_encdec import *
from utils import *
from nlp_tools import NLPTools


import time
import traceback
import os.path
import sys
import argparse
import cPickle
import logging
import pprint
import numpy
import collections
import signal
import math

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
logger = logging.getLogger(__name__)

def compute_stats_for_iteratior(iterator):
    stats_y = []
    stats_precomputed_x_features = []

    while True:
        batch = iterator.next() 
        if not batch:
            break

        logger.debug("Got batch %d,%d" % (batch['x_prev'].shape[1], batch['max_length']))
        
        x_data_prev = batch['x_prev']
        x_mask_prev = batch['x_mask_prev']
        x_data_next = batch['x_next']
        x_mask_next = batch['x_mask_next']
        x_precomputed_features = batch['x_precomputed_features']
        y_data = batch['y']
        y_data_prev = batch['y_prev']
        x_max_length = batch['max_length']

        max_length = batch['max_length']

        # Store features for computing mean and standard deviation of NLP features
        for i in range(y_data.shape[1]):
            stats_precomputed_x_features.append(x_precomputed_features[:, i])

        stats_precomputed_x_features_matrix = numpy.asarray(stats_precomputed_x_features)

        # Store the frequencies of speaker labels
        for i in range(y_data.shape[1]):
            stats_y.append(y_data[:, i])
        
        # print probabilities of all tokens
        stats_t = numpy.asarray(stats_y)

    print '# NLP Features'

    print 'stats_precomputed_x_features_matrix', stats_precomputed_x_features_matrix
    print 'stats_precomputed_x_features_matrix mean', numpy.mean(stats_precomputed_x_features_matrix, axis=0)
    print 'stats_precomputed_x_features_matrix std', numpy.std(stats_precomputed_x_features_matrix, axis=0)

    print 'Examples in total: ', stats_t.shape[0]

    print '# Speaker Classes (6-Way Speaker Classification)'
    print 'Class 1', numpy.where(stats_t[:,0] == 0)[0].shape[0]
    print 'Class 2', numpy.where(stats_t[:,0] == 5)[0].shape[0]
    print 'Class 3', numpy.where(stats_t[:,0] == 1)[0].shape[0]
    print 'Class 4', numpy.where(stats_t[:,0] == 2)[0].shape[0]
    print 'Class 5', numpy.where(stats_t[:,0] == 3)[0].shape[0]
    print 'Class 6', numpy.where(stats_t[:,0] == 4)[0].shape[0]

    print '# Auxiliary Tokens'
    print 'No auxiliary token', numpy.where(stats_t[:,1] == 0)[0].shape[0]
    print '<voice over>', numpy.where(stats_t[:,1] == 1)[0].shape[0]
    print '<off_screen>', numpy.where(stats_t[:,1] == 2)[0].shape[0]


def main(args):     
    logging.basicConfig(level = logging.DEBUG,
                        format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    logger.debug("Load model configuration")
    state = eval(args.prototype)() 
    model = DialogEncoderDecoder(state)
    rng = model.rng

    # Load data iterators
    logger.debug("Load data")
    sentence_break_symbols = [model.str_to_idx['.'], model.str_to_idx['?'], model.str_to_idx['!']]
    train_data, \
    valid_data, = get_train_iterator(state, sentence_break_symbols, False)
    test_data = get_test_iterator(state, sentence_break_symbols)

    # Force iterators to only loop once
    train_data.use_infinite_loop=False
    valid_data.use_infinite_loop=False
    test_data.use_infinite_loop=False

    tokens_to_predict_per_sample = 3

    # Start looping through the dataset
    start_time = time.time()
     
    train_cost = 0
    train_misclass_first = 0
    train_misclass_second = 0
    train_samples_done = 0 # Number of training examples done
    
    print '### Training Set'
    train_data.start()
    compute_stats_for_iteratior(train_data)

    print '### Validation Set'
    valid_data.start()
    compute_stats_for_iteratior(valid_data)

    print '### Test Set'
    test_data.start()
    compute_stats_for_iteratior(test_data)

    logger.debug("All done, exiting...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prototype", type=str, help="Use the model prototype", default='prototype_state')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    args = parse_args()
    main(args)
