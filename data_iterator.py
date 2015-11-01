import numpy as np
import theano
import theano.tensor as T
import sys, getopt
import logging

from state import *
from utils import *
from SS_dataset import *

import itertools
import sys
import pickle
import random
import datetime
import math
import copy
import collections

logger = logging.getLogger(__name__)

def remove_special_tokens(sentence, tokens_to_remove):
    # Remove special tokens, such as end-of-utterance and speaker tokens
    for token_to_remove in tokens_to_remove:
        while token_to_remove in sentence:
            sentence.remove(token_to_remove)

    return sentence

def create_samples(state, sentence_break_tokens, do_uniform_sampling, x):
    """
    This function takes an document (movie script) as input,
    and creates the (input, class) examples according to the state 
    and the list of end-of-sentence tokens.
    """
    
    # We store the examples in lists. X stores the sentences with one example per entry.
    X = []

    # Y store the classes. It is a list with one example per entry.
    # Inside, each entry is another list with (speaker class label, auxiliary label) . 
    # That is, the first entry of the inner list corresponds to the speaker classification label, 
    # where 0 = same speaker, 1 = <first_speaker>, 2 = s<econd_speaker>, 3 = <third_speaker>, 
    # 4 = <minor_speaker> and 5 = <pause>.
    # The second entry of the inner list corresponds to the auxiliary class, where
    # 0 = no auxiliary token, 1 = <voice_over> and 2 = <off_screen>

    Y = []
    # Y_Prev is similiar to Y, but stores the labels corresponding to the previous X, 
    # i.e. Y_Prev[i] stores the label for X[i-1]
    Y_Prev = []

    # Store document id
    Document_ID = []

    # Store NLP features
    Precomputed_Features = []

    # Define the list of tokens, which must be removed from the text
    tokens_to_remove = [state['eos_sym'], state['first_speaker_sym'], state['second_speaker_sym'], state['third_speaker_sym'], state['minor_speaker_sym'], state['voice_over_sym'], state['off_screen_sym'], state['pause_sym']]

    max_length = state['max_length']

    for idx in xrange(len(x[0])):

        # Token counter
        token_counter = collections.Counter(x[0][idx])

        # Insert punctuation sign before every </s> in dialogue
        eos_indices = numpy.where(numpy.asarray(x[0][idx]) == state['eos_sym'])[0]
        for index in reversed(range(eos_indices.shape[0])):
            if eos_indices[index] > 0:
                if x[0][idx][eos_indices[index]-1] not in sentence_break_tokens:
                    numpy.insert(x[0][idx], eos_indices[index]-1, sentence_break_tokens[0])

        # Insert sequence idx in a column of matrix X
        dialogue_length = len(x[0][idx])

        # Find all sentence break points
        sentence_break_positions = []
        for break_token in sentence_break_tokens:
            sentence_break_positions += list(numpy.where(numpy.asarray(x[0][idx]) == break_token)[0])

        # Find all end-of-dialogue tokens
        eod_indices = numpy.where(numpy.asarray(x[0][idx]) == state['eod_sym'])[0]

        # Define new list of sentence break points, but sorting away all sentences immediatly before or after an
        # end-of-dialogue token (since sentences across end-of-dialogue tokens should not be taken as examples)
        potential_indices = sentence_break_positions

        for index in reversed(range(len(potential_indices))):
            if potential_indices[index] == 0 or potential_indices[index] == dialogue_length - 1 \
                or x[0][idx][potential_indices[index]-1] == state['eod_sym'] \
                or x[0][idx][potential_indices[index]+1] == state['eod_sym']:
                    potential_indices = numpy.delete(potential_indices, index)
            
        # Now construct an example for every sentence break point (potential_indices)
        for index in range(len(potential_indices)):
            # First, we find the correct label for the sentence
            training_y = [0]
            
            if x[0][idx][potential_indices[index]+2] == state['first_speaker_sym']:
                training_y = [1]
            elif x[0][idx][potential_indices[index]+2] == state['second_speaker_sym']:
                training_y = [2]
            elif x[0][idx][potential_indices[index]+2] == state['third_speaker_sym']:
                training_y = [3]
            elif x[0][idx][potential_indices[index]+2] == state['minor_speaker_sym']:
                training_y = [4]
            elif x[0][idx][potential_indices[index]+2] == state['pause_sym']:
                training_y = [5]
            
            if training_y[0] > 0 and training_y[0] < 5:
                if x[0][idx][potential_indices[index]+3] == state['voice_over_sym']:
                    training_y += [1]
                elif x[0][idx][potential_indices[index]+3] == state['off_screen_sym']:
                    training_y += [2]
                else:
                    training_y += [0]
            else:
                training_y += [0]


            # Then we construct the sequence of words preceding and proceding the token
            start_pos = max(0, potential_indices[index]-max_length)
            # Cut of at nearest end-of-dialogue token
            previous_eod_tokens = numpy.where(eod_indices < potential_indices[index])[0]
            if len(previous_eod_tokens) > 0:
                start_pos = max(start_pos, eod_indices[previous_eod_tokens[-1]]+1)


            end_pos = min(dialogue_length, potential_indices[index]+max_length)

            # Cut of at nearest end-of-dialogue token
            next_eod_tokens = numpy.where(eod_indices > potential_indices[index])[0]
            if len(next_eod_tokens) > 0:
                end_pos = min(end_pos, eod_indices[next_eod_tokens[0]])


            prev_string = remove_special_tokens(x[0][idx][start_pos:potential_indices[index]], tokens_to_remove)
            next_string = remove_special_tokens(x[0][idx][potential_indices[index]+1:end_pos], tokens_to_remove)

            training_x = [prev_string, next_string]

            if index > 0:
                prev_sentence = remove_special_tokens(x[0][idx][max(start_pos, potential_indices[index-1]):potential_indices[index]], tokens_to_remove)
            else:
                prev_sentence = prev_string

            if index < len(potential_indices) - 1:
                next_sentence = remove_special_tokens(x[0][idx][potential_indices[index]+1:min(end_pos, potential_indices[index+1])], tokens_to_remove)
            else:
                next_sentence = next_string

            # If NLP features have been enabled, compute them now
            if state['use_precomputed_features']:
                precomputed = state['nlp_tools'].get_features(prev_sentence, next_sentence, token_counter)
                # If we wish to compute NLP features over more than the previous and next sentence, 
                # this line can be enabled. However, this didn't seem to improve performance.
                #precomputed = state['nlp_tools'].get_features(prev_string, next_string, token_counter)
            else:
                precomputed = numpy.zeros((state['precomputed_features_count']), dtype='float32')

            # If data example contains at least one word in the previous and next sentence,
            # add it as a data example
            if (potential_indices[index] - start_pos) >= 0 and (end_pos - potential_indices[index] + 1) >= 0:
                # Keep track of previous label
                if len(Y) > 0:
                    Y_Prev.append(Y[-1])
                else: # If there was no previous label, assume it is a "minor speaker" with no auxiliary label
                    Y_Prev.append([4,0])

                Y.append(training_y)
                X.append(training_x)
                Precomputed_Features.append(precomputed)
                Document_ID.append(idx)


    # If uniform sampling has been enabled, then resample with replacement from original dataset
    # with uniform probabilities for each class. 
    # This may help the neural networks to learn better representations, 
    # in cases where the classes are very unbalanced/
    if do_uniform_sampling:
        Resampled_X = []
        Resampled_Y = []
        Resampled_Y_Prev = []
        Resampled_Precomputed_Features = []
        Resampled_Document_ID = []
        for i in range(len(X)):
            c = random.randint(0, 5)
            idx = -1
            for i in range(len(X)):
                if Y[i][0] == c:
                    idx = i
                    break

            if idx >= 0:
                Resampled_X.append(X[idx])
                Resampled_Y.append(Y[idx])
                Resampled_Y_Prev.append(Y_Prev[idx])
                Resampled_Precomputed_Features.append(Precomputed_Features[idx])
                Resampled_Document_ID.append(Document_ID[idx])

                # Move sample to end of X, Y and Precomputed_Features
                del X[idx]
                del Y[idx]
                del Y_Prev[idx]
                del Document_ID[idx]
                del Precomputed_Features[idx]
                X.append(Resampled_X[-1])
                Y.append(Resampled_Y[-1])
                Y_Prev.append(Resampled_Y_Prev[-1])
                Precomputed_Features.append(Resampled_Precomputed_Features[-1])
                Document_ID.append(Resampled_Document_ID[-1])

        # Overwrite previous variables
        X = Resampled_X
        Y = Resampled_Y
        Y_Prev = Resampled_Y_Prev
        Precomputed_Features = Resampled_Precomputed_Features
        Document_ID = Resampled_Document_ID

    # Finally, convert all lists into numpy arrays so that Theano can read them, and return the examples.
    X_Prev = numpy.zeros((max_length, len(X)), dtype='int32')
    X_Next = numpy.zeros((max_length, len(X)), dtype='int32')
    X_Mask_Prev = numpy.zeros((max_length, len(X)), dtype='float32')
    X_Mask_Next = numpy.zeros((max_length, len(X)), dtype='float32')

    X_Precomputed_Features = numpy.zeros((state['precomputed_features_count'], len(X)), dtype='float32')

    Y_Final = numpy.zeros((2, len(X)), dtype='int32')
    Y_Prev_Final = numpy.zeros((2, len(X)), dtype='int32')

    Document_ID_Final = numpy.zeros((len(X)), dtype='int32')

    for idx in range(len(X)):
        X_Prev[:len(X[idx][0]), idx] = numpy.asarray(X[idx][0])
        X_Next[:len(X[idx][1]), idx] = numpy.asarray(X[idx][1])
        X_Mask_Prev[:len(X[idx][0]), idx] = 1
        X_Mask_Next[:len(X[idx][1]), idx] = 1

        X_Precomputed_Features[:, idx] = Precomputed_Features[idx]

        Y_Final[0, idx] = Y[idx][0]
        Y_Final[1, idx] = Y[idx][1]

        Y_Prev_Final[0, idx] = Y_Prev[idx][0]
        Y_Prev_Final[1, idx] = Y_Prev[idx][1]

        Document_ID_Final[idx] = Document_ID[idx] 

    return {'x_prev': X_Prev,                                 \
            'x_mask_prev': X_Mask_Prev,                       \
            'x_next': X_Next,                                 \
            'x_mask_next': X_Mask_Next,                       \
            'x_precomputed_features': X_Precomputed_Features, \
            'y': Y_Final,                                     \
            'y_prev': Y_Prev_Final,                           \
            'document_id': Document_ID_Final,                 \
            'max_length': max_length                          \
           }

class Iterator(SSIterator):
    def __init__(self, dialogue_file, batch_size, sentence_break_tokens, **kwargs):
        SSIterator.__init__(self, dialogue_file, batch_size,                 \
                            sentence_break_tokens,                          \
                            max_len=kwargs.pop('max_len', -1),               \
                            use_infinite_loop=kwargs.pop('use_infinite_loop', False))

        # TODO: max_len should be handled here and SSIterator should zip semantic_data and 
        # data. 
        self.k_batches = kwargs.pop('sort_k_batches', 20)
        # TODO: For backward compatibility. This should be removed in future versions
        # i.e. remove all the x_reversed computations in the model itself.
        self.state = kwargs.pop('state', None)
        # ---------------- 
        self.batch_iter = None
        self.sentence_break_tokens = sentence_break_tokens
        self.uniform_sampling_across_classes = kwargs.pop('uniform_sampling_across_classes', None)

    def get_homogenous_batch_iter(self, batch_size = -1):
        while True:
            batch_size = self.batch_size if (batch_size == -1) else batch_size 
           
            data = []
            for k in range(self.k_batches):
                batch = SSIterator.next(self)
                if batch:
                    data.append(batch)
            
            if not len(data):
                return
            
            number_of_batches = len(data)
            data = list(itertools.chain.from_iterable(data))

            # Split list of words from the dialogue index
            data_x = []
            data_semantic = []
            for i in range(len(data)):
                data_x.append(data[i][0])
                data_semantic.append(data[i][1])

            x = numpy.asarray(list(itertools.chain(data_x)))

            lens = numpy.asarray([map(len, x)])
            order = numpy.argsort(lens.max(axis=0))
                 
            for k in range(number_of_batches):
                indices = order[k * batch_size:(k + 1) * batch_size]
                if self.uniform_sampling_across_classes > 0:
                    full_samples_set = create_samples(self.state, self.sentence_break_tokens, True, [x[indices]])
                    self.uniform_sampling_across_classes = self.uniform_sampling_across_classes - int(math.ceil(float(full_samples_set['y'].shape[1]) / float(self.state['bs'])))
                else:
                    full_samples_set = create_samples(self.state, self.sentence_break_tokens, False, [x[indices]])

                # Split the oversized batch
                splits = int(math.ceil(float(full_samples_set['y'].shape[1]) / float(self.state['bs'])))
                batches = []
                for i in range(0, splits):
                    start_index = i*self.state['bs']
                    end_index = min((i+1)*self.state['bs'], full_samples_set['y'].shape[1])

                    batch = {'x_prev': full_samples_set['x_prev'][:,start_index:end_index],           \
                             'x_mask_prev': full_samples_set['x_mask_prev'][:,start_index:end_index], \
                             'x_next': full_samples_set['x_next'][:,start_index:end_index],           \
                             'x_mask_next': full_samples_set['x_mask_next'][:,start_index:end_index], \
                             'x_precomputed_features': full_samples_set['x_precomputed_features'][:,start_index:end_index],  \
                             'y': full_samples_set['y'][:,start_index:end_index],                     \
                             'y_prev': full_samples_set['y_prev'][:,start_index:end_index],           \
                             'max_length': full_samples_set['max_length'],                            \
                             'document_id': full_samples_set['document_id'][start_index:end_index],   \
                             'num_samples': (end_index-start_index)                                   \
                             }

                    batches.append(batch)

                for batch in batches:
                    if batch:
                        yield batch



    def start(self):
        SSIterator.start(self)
        self.batch_iter = None

    def next(self, batch_size = -1):
        """ 
        We can specify a batch size,
        independent of the object initialization. 
        """
        if not self.batch_iter:
            self.batch_iter = self.get_homogenous_batch_iter(batch_size)
        try:
            batch = next(self.batch_iter)
        except StopIteration:
            return None
        return batch

def get_train_iterator(state, sentence_break_tokens, uniform_sampling_across_classes):    
    train_data = Iterator(
        state['train_dialogues'],
        int(state['bs']),
        sentence_break_tokens,
        state=state,
        seed=state['seed'],
        use_infinite_loop=True, 
        max_len=-1,
        uniform_sampling_across_classes=uniform_sampling_across_classes)

    # Check if we are in testing mode or not
    valid_data = Iterator(
        state['valid_dialogues'],
        int(state['bs']),
        sentence_break_tokens,
        state=state,
        seed=state['seed'],
        use_infinite_loop=False,
        max_len=-1,
        uniform_sampling_across_classes=0)

    return train_data, valid_data 

def get_test_iterator(state, sentence_break_tokens):
    assert 'test_dialogues' in state
    test_path = state.get('test_dialogues')
    semantic_test_path = state.get('test_semantic', None)

    test_data = Iterator(
        test_path,
        int(state['bs']),
        sentence_break_tokens,
        state=state,
        seed=state['seed'],
        use_infinite_loop=False,
        max_len=-1,
        uniform_sampling_across_classes=0)
    return test_data
