# -*- coding: utf-8 -*-
#!/usr/bin/env python

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


import matplotlib
matplotlib.use('Agg')
import pylab


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

### Unique RUN_ID for this execution
RUN_ID = str(time.time())

### Performance measures are defined here
measures = ["train_cost", "train_misclass_first", "train_misclass_second", "valid_cost", "valid_misclass_first", "valid_misclass_second", "valid_emi", "valid_bleu_n_1", "valid_bleu_n_2", "valid_bleu_n_3", "valid_bleu_n_4", 'valid_jaccard', 'valid_recall_at_1', 'valid_recall_at_5', 'valid_mrr_at_5', 'tfidf_cs_at_1', 'tfidf_cs_at_5']

def init_timings():
    timings = {}
    for m in measures:
        timings[m] = []
    return timings

def save(model, timings, post_fix = ''):
    print "Saving the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    model.save(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'model.npz')
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'state.pkl', 'w'))
    numpy.savez(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'timing.npz', **timings)
    signal.signal(signal.SIGINT, s)
    
    print "Model saved, took {}".format(time.time() - start)

def load(model, filename):
    print "Loading the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model.load(filename)
    signal.signal(signal.SIGINT, s)

    print "Model loaded, took {}".format(time.time() - start)

def main(args):     
    logging.basicConfig(level = logging.DEBUG,
                        format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")
     
    state = eval(args.prototype)() 
    timings = init_timings()

    # Resume model if specified by command line arguments.
    if args.resume != "":
        # Load in state and timings file for model to resume
        logger.debug("Resuming %s" % args.resume)
        
        state_file = args.resume + '_state.pkl'
        timings_file = args.resume + '_timing.npz'
        
        if os.path.isfile(state_file) and os.path.isfile(timings_file):
            logger.debug("Loading previous state")
            
            state = cPickle.load(open(state_file, 'r'))
            timings = dict(numpy.load(open(timings_file, 'r')))
            for x, y in timings.items():
                timings[x] = list(y)
        else:
            raise Exception("Cannot resume, cannot find files!")

    logger.debug("State:\n{}".format(pprint.pformat(state)))
    logger.debug("Timings:\n{}".format(pprint.pformat(timings)))

    # If user specified --force_train_all_wordemb, the model will train all word embeddings,
    # regardless of any other model configuration variables
    if args.force_train_all_wordemb == True:
        state['fix_pretrained_word_embeddings'] = False

    model = DialogEncoderDecoder(state)
    rng = model.rng

    if args.resume != "":
        # Load in model parameters for model to resume
        filename = args.resume + '_model.npz'
        if os.path.isfile(filename):
            logger.debug("Loading previous model")
            load(model, filename)
        else:
            raise Exception("Cannot resume, cannot find model file!")
        
        if 'run_id' not in model.state:
            raise Exception('Backward compatibility not ensured! (need run_id in state)')           

    else:
        # assign new run_id key
        model.state['run_id'] = RUN_ID

    # Compile Theano functions
    logger.debug("Compile trainer")
    logger.debug("Training with exact log-likelihood")
    train_batch = model.build_train_function()

    eval_batch = model.build_eval_function()

    logger.debug("Load data")
    sentence_break_symbols = [model.str_to_idx['.'], model.str_to_idx['?'], model.str_to_idx['!']]
    train_data, \
    valid_data, = get_train_iterator(state, sentence_break_symbols, args.uniform_sampling_across_classes)
    train_data.start()

    # The model is always predicting the speaker class, change-of-turn / no-change-of-turn class and auxiliary class
    tokens_to_predict_per_sample = 3

    # Start looping through the dataset
    step = 0
    patience = state['patience'] 
    start_time = time.time()
     
    train_cost = 0
    train_misclass_first = 0 # Number of misclassifications of first class type (speaker class)
    train_misclass_second = 0 # Number of misclassifications of second class type (turn-taking class)
    train_samples_done = 0 # Number of training examples done
    
    # Start training loop
    while (step < state['loop_iters'] and
            (time.time() - start_time)/60. < state['time_stop'] and
            patience >= 0):

        # Training phase
        batch = train_data.next() 

        # Train finished
        if not batch:
            # Restart training
            logger.debug("Got None...")
            break
        
        # Retrieve variables from batch
        logger.debug("[TRAIN] - Got batch %d,%d" % (batch['x_prev'].shape[1], batch['max_length']))

        x_data_prev = batch['x_prev']
        x_mask_prev = batch['x_mask_prev']
        x_data_next = batch['x_next']
        x_mask_next = batch['x_mask_next']
        x_precomputed_features = batch['x_precomputed_features']
        y_data = batch['y']
        y_data_prev = batch['y_prev']
        x_max_length = batch['max_length']

        max_length = batch['max_length']

        # Train on batch
        c, miscl_first, miscl_second = train_batch(x_data_prev, x_mask_prev, x_data_next, x_mask_next, x_precomputed_features, y_data, y_data_prev, x_max_length)

        # Keep track of log-likelihood and misclassifications
        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN cost .. skipping")
            continue

        train_cost += c
        train_misclass_first += miscl_first
        train_misclass_second += miscl_second


        train_samples_done += batch['num_samples']

        this_time = time.time()

        # Print training statistics
        if step % state['train_freq'] == 0:
            elapsed = this_time - start_time
            h, m, s = ConvertTimedelta(this_time - start_time)

            print ".. %.2d:%.2d:%.2d %4d mb # %d bs %d maxl %d acc_cost = %.4f acc_word_perplexity = %.4f acc_mean_error_turn_taking_class = %.4f acc_mean_error_speaker_class = %.4f" % (h, m, s,\
                             state['time_stop'] - (time.time() - start_time)/60.,\
                             step, \
                             batch['x_prev'].shape[1], \
                             batch['max_length'], \
                             float((train_cost/train_samples_done)/tokens_to_predict_per_sample), \
                             math.exp(float((train_cost/train_samples_done)/tokens_to_predict_per_sample)), \
                             float(train_misclass_first/float(train_samples_done)), \
                             float(train_misclass_second/float(train_samples_done)))


        # Start validation loop
        if valid_data is not None and\
            step % state['valid_freq'] == 0 and step > 1:
                valid_data.start()
                valid_cost = 0
                valid_misclass_first = 0
                valid_misclass_second = 0
                valid_samples_done = 0

                # Prepare variables for plotting histogram over word-perplexities and mutual information
                valid_data_len = valid_data.data_len

                logger.debug("[VALIDATION START]") 
                
                while True:
                    batch = valid_data.next()
                    # Train finished
                    if not batch:
                        break
                     
                    logger.debug("[VALID] - Got batch %d,%d" % (batch['x_prev'].shape[1], batch['max_length']))

                    x_data_prev = batch['x_prev']
                    x_mask_prev = batch['x_mask_prev']
                    x_data_next = batch['x_next']
                    x_mask_next = batch['x_mask_next']
                    x_precomputed_features = batch['x_precomputed_features']
                    y_data = batch['y']
                    y_data_prev = batch['y_prev']


                    x_max_length = batch['max_length']
                    c, _, miscl_first, miscl_second, _, _ = eval_batch(x_data_prev, x_mask_prev, x_data_next, x_mask_next, x_precomputed_features, y_data, y_data_prev, x_max_length)

                    if numpy.isinf(c) or numpy.isnan(c):
                        continue
                    
                    valid_cost += c
                    valid_misclass_first += miscl_first
                    valid_misclass_second += miscl_second

                    valid_samples_done += batch['num_samples']

                logger.debug("[VALIDATION END]") 
                
                valid_cost /= float(valid_samples_done*tokens_to_predict_per_sample)
                valid_misclass_first /= float(valid_samples_done)
                valid_misclass_second /= float(valid_samples_done)

                if len(timings["valid_cost"]) == 0 or valid_cost < numpy.min(timings["valid_cost"]):
                    patience = state['patience']
                    # Saving model if decrease in validation cost
                    save(model, timings)
                elif valid_cost >= timings["valid_cost"][-1] * state['cost_threshold']:
                    patience -= 1

                if args.save_every_valid_iteration:
                    save(model, timings, '_' + str(step) + '_')



                print "** valid cost (NLL) = %.4f, valid word-perplexity = %.4f, valid mean turn-taking class error = %.4f, valid mean speaker class error = %.4f, patience = %d" % (float(valid_cost), float(math.exp(valid_cost)), float(valid_misclass_first), float(valid_misclass_second), patience)


                timings["train_cost"].append((train_cost/train_samples_done)/tokens_to_predict_per_sample)
                timings["train_misclass_first"].append(float(train_misclass_first)/float(train_samples_done))
                timings["train_misclass_second"].append(float(train_misclass_second)/float(train_samples_done))
                timings["valid_cost"].append(valid_cost)
                timings["valid_misclass_first"].append(float(valid_misclass_first))
                timings["valid_misclass_second"].append(float(valid_misclass_second))

                # Reset train cost, train misclass and train done
                train_cost = 0
                train_misclass_first = 0
                train_misclass_second = 0
                train_samples_done = 0

        step += 1

    logger.debug("All done, exiting...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")
    parser.add_argument("--force_train_all_wordemb", action='store_true', help="If true, will force the model to train all word embeddings in the encoder. This switch can be used to fine-tune a model which was trained with fixed (pretrained)  encoder word embeddings.")
    parser.add_argument("--save_every_valid_iteration", action='store_true', help="If true, will save a copy of the model at every validation iteration.")
    parser.add_argument("--uniform_sampling_across_classes", type=int, default="0", help="The number of batches the data iterator will sample uniformly across speaker classes. After this number of batches, the data iterator will startt to sample clases proportional to their real normalized frequency in the training set.")
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='prototype_state')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Example run:
    #    THEANO_FLAGS=mode=FAST_COMPILE,floatX=float32 python train.py --prototype prototype_movies

    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    args = parse_args()
    main(args)
