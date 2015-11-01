#!/usr/bin/env python
"""
Evaluation script.


Run example:

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,allow_gc=True,scan.allow_gc=False,nvcc.flags=-use_fast_math python evaluate.py Output/1432724394.9_MovieScriptModel --document_ids Data/Test_Shuffled_Dataset_Labels.txt &> Test_Eval_Output.txt

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,allow_gc=True,scan.allow_gc=False,nvcc.flags=-use_fast_math python evaluate.py Output/1443431225.78_MovieScriptModel

"""

import argparse
import cPickle
import traceback
import logging
import time
import sys

import os
import numpy
import codecs
import math

from dialog_encdec import DialogEncoderDecoder 
from numpy_compat import argpartition
from state import * 
from data_iterator import get_test_iterator

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
    
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")
    
    parser.add_argument("--test-path",
            type=str, help="File of test data")

    parser.add_argument("--document-ids",
            type=str, help="File containing document ids for each example (one id per line, if there are multiple tabs the first entry will be taken as the doc id). If this is given, the script will compute standard deviations across documents for all metrics. Currently this is not implemented.")

    return parser.parse_args()

def main():
    args = parse_args()
    state = prototype_state()
   
    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"

    with open(state_path) as src:
        state.update(cPickle.load(src)) 
    
    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    # Force batch size to be one, so that we can condition the prediction at time t on its prediction at time t-1.
    state['bs'] = 1 
 
    model = DialogEncoderDecoder(state)
    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")
    
    eval_batch = model.build_eval_function()
    
    if args.test_path:
        state['test_dialogues'] = args.test_path

    sentence_break_symbols = [model.str_to_idx['.'], model.str_to_idx['?'], model.str_to_idx['!']]
    test_data = get_test_iterator(state, sentence_break_symbols)
    test_data.start()

    tokens_per_sample = 3

    # Load document ids
    if args.document_ids:
        print("Warning. Evaluation using document ids is not supported")
        labels_file = open(args.document_ids, 'r')
        labels_text = labels_file.readlines()
        document_ids = numpy.zeros((len(labels_text)), dtype='int32')
        for i in range(len(labels_text)):
            document_ids[i] = int(labels_text[i].split('\t')[0])

        unique_document_ids = numpy.unique(document_ids)
        
        assert(test_data.data_len == document_ids.shape[0])

    else:
        document_ids = numpy.zeros((test_data.data_len), dtype='int32')
        unique_document_ids = numpy.unique(document_ids)
    
    # Variables to store test statistics
    test_cost = 0 # negative log-likelihood
    test_misclass_first = 0 # misclassification error-rate
    test_misclass_second = 0 # misclassification error-rate
    test_samples_done = 0 # number of examples evaluated

    # Number of examples in dataset
    test_data_len = test_data.data_len

    logger.debug("[TEST START]") 

    prev_doc_id = -1
    prev_predicted_speaker = 4

    while True:
        batch = test_data.next()
        # Train finished
        if not batch:
            break
         
        logger.debug("[TEST] - Got batch %d,%d" % (batch['x_prev'].shape[1], batch['max_length']))

        x_data_prev = batch['x_prev']
        x_mask_prev = batch['x_mask_prev']
        x_data_next = batch['x_next']
        x_mask_next = batch['x_mask_next']
        x_precomputed_features = batch['x_precomputed_features']
        y_data = batch['y']
        y_data_prev_true = batch['y_prev']
        x_max_length = batch['max_length']


        doc_id = batch['document_id'][0]
        y_data_prev_estimate = numpy.zeros((2, 1), dtype='int32')
        # If we continue in the same dialogue, use previous prediction to inform current prediction
        if prev_doc_id == doc_id:
            y_data_prev_estimate[0,0] = prev_predicted_speaker
        else: # Otherwise, we assume the previous (non-existing utterance) was labelled as "minor_speaker"
            y_data_prev_estimate[0,0] = 4

        #print 'y_data_prev_estimate', y_data_prev_estimate
        #print 'y_data_prev_true', y_data_prev_true

        c, _, miscl_first, miscl_second, training_preds_first, training_preds_second = eval_batch(x_data_prev, x_mask_prev, x_data_next, x_mask_next, x_precomputed_features, y_data, y_data_prev_estimate, x_max_length)

        prev_doc_id = doc_id
        prev_predicted_speaker = training_preds_second[0]

        test_cost += c
        test_misclass_first += miscl_first
        test_misclass_second += miscl_second
        test_samples_done += batch['num_samples']
     
    logger.debug("[TEST END]") 

    test_cost /= float(test_samples_done*tokens_per_sample)
    test_misclass_first /= float(test_samples_done)
    test_misclass_second /= float(test_samples_done)

    print "** test cost (NLL) = %.4f, valid word-perplexity = %.4f, valid mean turn-taking class error = %.4f, valid mean speaker class error = %.4f" % (float(test_cost), float(math.exp(test_cost)), float(test_misclass_first), float(test_misclass_second))


    logger.debug("All done, exiting...")

if __name__ == "__main__":
    main()
