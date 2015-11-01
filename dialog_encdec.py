"""
Dialog segmentation model based on GRU RNNs.

The model learns to predict the speaker segmentation from dialogue: 
- speaker changes and speaker identity: <first_speaker>, <second_speaker>, <third_speaker> and <minor_speaker>,
          where it is assumed that the first three speakers are sorted by frequency (e.g. so that <first_speaker> is
          the unique speaker with the most utterances), and that <minor_speaker> covers all other speakers.
- precomputed segmentation specific to movies: <voice_over> and <off_screen>. These can safely be for non-movie data.

The input to the model is expected to be two sequences of tokens (words), one sequence before the segmentation label and one sequence after the segmentation label. Each sequence is processed by a separate GRU RNN. The final RNN hidden states are then concatenated and given as input to a single-layer MLP function. Finally, the MLP output is transformed with a softmax function to give probabilities over speaker and precomputed segmentation.

The code is inspired from the hed-dlg hierarchical encoder-decoder architecture found at:
https://github.com/sordonia/hed-dlg

"""
__docformat__ = 'restructedtext en'
__authors__ = ("Iulian Vlad Serban")

import theano
import theano.tensor as T
import numpy as np
import cPickle
import logging
logger = logging.getLogger(__name__)

from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv3d2d import *
from collections import OrderedDict

from model import *
from utils import *

import operator

# Theano speed-up
theano.config.scan.allow_gc = False

def relu(x):
    return T.switch(x<0, 0, x)

def add_to_params(params, new_param):
    params.append(new_param)
    return new_param

class EncoderDecoderBase():
    def __init__(self, state, rng, parent):
        self.rng = rng
        self.parent = parent
        
        self.state = state
        self.__dict__.update(state)
        
        self.rec_activation = eval(self.rec_activation)
         
        self.params = []

# This is an encoder RNN class, which maps a sequence of tokens into a hidden state.
class TextEncoder(EncoderDecoderBase):
    def init_params(self, word_embedding_param):
        # Initialzie W_emb to given word embeddings
        assert(word_embedding_param != None)
        self.W_emb = word_embedding_param

        """ sent weights """
        self.h_initial = add_to_params(self.params, theano.shared(value=np.zeros((1, self.qdim), dtype='float32'), name='h_initial'+self.name))
        self.W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in'+self.name))
        self.W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh'+self.name))
        self.b_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_hh'+self.name))
        
        if self.sent_step_type == "gated":
            self.W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_r'+self.name))
            self.W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_z'+self.name))
            self.W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_r'+self.name))
            self.W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_z'+self.name))
            self.b_z = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_z'+self.name))
            self.b_r = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_r'+self.name))

    def approx_embedder(self, x):
        return self.W_emb[x]

    def plain_sent_step(self, x_t, m_t, *args):
        args = iter(args)
        h_tm1 = next(args)

        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
         
        # Use mask to decide when to stop computing forward pass.
        h_t = (1 - m_t) * h_tm1 \
              + m_t * self.rec_activation(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W_hh) + self.b_hh)

        return [h_t]

    def gated_sent_step(self, x_t, m_t, *args):
        args = iter(args)
        h_tm1 = next(args)

        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x') 

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_r) + T.dot(h_tm1, self.W_hh_r) + self.b_r)
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_z) + T.dot(h_tm1, self.W_hh_z) + self.b_z)
        h_tilde = self.rec_activation(T.dot(x_t, self.W_in) + T.dot(r_t * h_tm1, self.W_hh) + self.b_hh)

        # Use mask to decide when to stop computing forward pass.
        h_t = (1 - m_t) * h_tm1 + m_t * ((np.float32(1.0) - z_t) * h_tm1 + z_t * h_tilde)
        
        # return both reset state and non-reset state
        return [h_t, r_t, z_t, h_tilde]

    def build_encoder(self, x, xmask, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True
         
        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1

        # if it is not one_step then we initialize everything to previous state or intial state  
        if not one_step:
            h_0 = T.repeat(self.h_initial, batch_size, axis=0)

        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_h' in kwargs 
            h_0 = kwargs['prev_h']

        xe = self.approx_embedder(x)

        ones_scalar = theano.shared(value=numpy.ones((1), dtype='float32'), name='ones_scalar')

        # Gated Encoder
        if self.sent_step_type == "gated":
            f_enc = self.gated_sent_step
            o_enc_info = [h_0, None, None, None]

        else:
            f_enc = self.plain_sent_step
            o_enc_info = [h_0]


        # Run through all the sentence (encode everything)
        if not one_step: 
            _res, _ = theano.scan(f_enc,
                              sequences=[xe, xmask],\
                              outputs_info=o_enc_info)
        else: # Make just one step further
            _res = f_enc(xe, xmask, [h_0])[0]

        # Get the hidden state sequence
        h = _res[0]
        return h

    def __init__(self, state, rng, word_embedding_param, parent, name):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.name = name
        self.init_params(word_embedding_param)

# This is a decoder class, which outputs probabilities over different classes.
class TextDecoder(EncoderDecoderBase):
    EVALUATION = 1
    SAMPLING = 2
    BEAM_SEARCH = 3

    def __init__(self, state, rng, parent):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.trng = MRG_RandomStreams(self.seed)
        self.init_params()

    def init_params(self): 
        if self.multiplicative_input_from_encoders:
            if self.bidirectional_encoder:
                self.input_dim = self.qdim*2
            else:
                self.input_dim = self.qdim
        else:
            if self.bidirectional_encoder:
                self.input_dim = self.qdim*4
            else:
                self.input_dim = self.qdim*2

        if self.use_precomputed_features:
            self.input_dim += self.precomputed_features_count

        """ Decoder weights """
        self.Wd_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.input_dim, self.mlp_out_dim), name='Wd_in'))
        self.bd_in = add_to_params(self.params, theano.shared(value=np.zeros((self.mlp_out_dim,), dtype='float32'), name='bd_in')) 


        if self.condition_on_previous_speaker_class:
            self.Wd_softmax_first = add_to_params(self.params, theano.shared(value=NormalInit3D(self.rng, self.segmentation_token_count, self.mlp_out_dim, self.segmentation_token_count), name='Wd_softmax_first'))
            self.bd_softmax_first = add_to_params(self.params, theano.shared(value=np.zeros((self.segmentation_token_count, self.segmentation_token_count), dtype='float32'), name='bd_softmax__first'))

            self.Wd_softmax_second = add_to_params(self.params, theano.shared(value=NormalInit3D(self.rng, self.segmentation_token_count, self.mlp_out_dim, self.segmentation_token_count), name='Wd_softmax_second'))
            self.bd_softmax_second = add_to_params(self.params, theano.shared(value=np.zeros((self.segmentation_token_count, self.segmentation_token_count), dtype='float32'), name='bd_softmax__second'))

            self.Wd_softmax_third = add_to_params(self.params, theano.shared(value=NormalInit3D(self.rng, self.segmentation_token_count, self.mlp_out_dim, self.segmentation_token_count), name='Wd_softmax_third'))
            self.bd_softmax_third = add_to_params(self.params, theano.shared(value=np.zeros((self.segmentation_token_count, self.segmentation_token_count), dtype='float32'), name='bd_softmax__third'))
        else:
            self.Wd_softmax_first = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.mlp_out_dim, self.segmentation_token_count), name='Wd_softmax_first'))
            self.bd_softmax_first = add_to_params(self.params, theano.shared(value=np.zeros((self.segmentation_token_count,), dtype='float32'), name='bd_softmax__first'))

            self.Wd_softmax_second = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.mlp_out_dim, self.segmentation_token_count), name='Wd_softmax_second'))
            self.bd_softmax_second = add_to_params(self.params, theano.shared(value=np.zeros((self.segmentation_token_count,), dtype='float32'), name='bd_softmax__second'))

            self.Wd_softmax_third = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.mlp_out_dim, self.segmentation_token_count), name='Wd_softmax_third'))
            self.bd_softmax_third = add_to_params(self.params, theano.shared(value=np.zeros((self.segmentation_token_count,), dtype='float32'), name='bd_softmax__third'))


    def build_next_probs_predictor(self, inp, x, prev_state):
        """ 
        Return output probabilities given prev_words x, hierarchical pass hs, and previous hd
        hs should always be the same (and should not be updated).
        """
        return self.build_decoder(inp, x, mode=TextDecoder.BEAM_SEARCH, prev_state=prev_state)
    
    def build_decoder(self, decoder_inp, y=None, y_prev=None, mode=EVALUATION):
        # Run the decoder

        if self.mlp_activation_function == 'tanh':
            hidden_activation = T.tanh(T.dot(decoder_inp, self.Wd_in) + self.bd_in)
        elif self.mlp_activation_function == 'rectifier':
            hidden_activation = relu(T.dot(decoder_inp, self.Wd_in) + self.bd_in)
        elif self.mlp_activation_function == 'linear':
            hidden_activation = T.dot(decoder_inp, self.Wd_in) + self.bd_in
        else:
            raise Exception("Invalid activation function specified for MLP!") 

        if self.condition_on_previous_speaker_class:
                first_output = T.nnet.softmax(T.dot(hidden_activation, self.Wd_softmax_first[y_prev[0]][0,:,:]) + self.bd_softmax_first[y_prev[0]])

                second_output = T.nnet.softmax(T.dot(hidden_activation, self.Wd_softmax_second[y_prev[0]][0,:,:]) + self.bd_softmax_second[y_prev[0]])
                third_output = T.nnet.softmax(T.dot(hidden_activation, self.Wd_softmax_third[y_prev[0]][0,:,:]) + self.bd_softmax_third[y_prev[0]])

                outputs = T.concatenate([first_output, second_output, third_output])

        else:
                first_output = T.nnet.softmax(T.dot(hidden_activation, self.Wd_softmax_first) + self.bd_softmax_first)
                second_output = T.nnet.softmax(T.dot(hidden_activation, self.Wd_softmax_second) + self.bd_softmax_second)
                third_output = T.nnet.softmax(T.dot(hidden_activation, self.Wd_softmax_third) + self.bd_softmax_third)
                outputs = T.concatenate([first_output, second_output, third_output])

        # EVALUATION  / BEAM SEARCH: Return outputs
        if mode == TextDecoder.EVALUATION:
            first_target_outputs = GrabProbs(first_output, y[0])
            second_target_outputs = GrabProbs(second_output, y[1])
            third_target_outputs = GrabProbs(third_output, y[1])
            target_outputs = T.concatenate([first_target_outputs, second_target_outputs, third_target_outputs])

            return outputs, target_outputs
        elif mode == TextDecoder.BEAM_SEARCH:
            return outputs
        # SAMPLING    : Return a vector with sample
        elif mode == TextDecoder.SAMPLING:
            first_sample = self.trng.multinomial(pvals=first_output, dtype='int64').argmax(axis=-1)
            second_sample = self.trng.multinomial(pvals=second_output, dtype='int64').argmax(axis=-1)
            third_sample = self.trng.multinomial(pvals=third_output, dtype='int64').argmax(axis=-1)
            return T.concatenate([first_sample, second_sample, third_sample])



class DialogEncoderDecoder(Model):
    def indices_to_words(self, seq, exclude_end_sym=True):
        """
        Converts a list of words to a list
        of word ids. Use unk_sym if a word is not
        known.
        """
        def convert():
            for word_index in seq:
                if word_index > len(self.idx_to_str):
                    raise ValueError('Word index is too large for the model vocabulary!')
                if not exclude_end_sym or (word_index != self.eos_sym):
                    yield self.idx_to_str[word_index]
        return list(convert())

    def words_to_indices(self, seq):
        """
        Converts a list of words to a list
        of word ids. Use unk_sym if a word is not
        known.
        """
        return [self.str_to_idx.get(word, self.unk_sym) for word in seq]

    def compute_updates(self, training_cost, params):
        updates = []
         
        grads = T.grad(training_cost, params)
        grads = OrderedDict(zip(params, grads))

        # Clip stuff
        c = numpy.float32(self.cutoff)
        clip_grads = []
        
        norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
        normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float32(1.))
        notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
         
        for p, g in grads.items():
            clip_grads.append((p, T.switch(notfinite, numpy.float32(.1) * p, g * normalization)))
        
        grads = OrderedDict(clip_grads)

        if self.initialize_from_pretrained_word_embeddings and self.fix_pretrained_word_embeddings:
            # Keep pretrained word embeddings fixed
            logger.debug("Will use mask to fix pretrained word embeddings")
            grads[self.W_emb] = grads[self.W_emb] * self.W_emb_pretrained_mask
        else:
            logger.debug("Will train all word embeddings")

        if self.updater == 'adagrad':
            updates = Adagrad(grads, self.lr)  
        elif self.updater == 'sgd':
            raise Exception("Sgd not implemented!")
        elif self.updater == 'adadelta':
            updates = Adadelta(grads)
        elif self.updater == 'rmsprop':
            updates = RMSProp(grads, self.lr)
        elif self.updater == 'adam':
            updates = Adam(grads)
        else:
            raise Exception("Updater not understood!") 

        return updates
  
    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            # Compile functions
            logger.debug("Building train function")
                
            self.train_fn = theano.function(inputs=[self.x_data_prev, self.x_mask_prev, 
                                                         self.x_data_next, self.x_mask_next, 
                                                         self.x_precomputed_features, self.y_data, 
                                                         self.y_data_prev, self.x_max_length],
                                            outputs=[self.training_cost, self.training_misclassification_first_acc, self.training_misclassification_second_acc],
                                            updates=self.updates, 
                                            on_unused_input='warn', 
                                            name="train_fn")

        return self.train_fn

    def build_eval_function(self):
        if not hasattr(self, 'eval_fn'):
            # Compile functions
            logger.debug("Building evaluation function")
            self.eval_fn = theano.function(inputs=[self.x_data_prev, self.x_mask_prev, 
                                                         self.x_data_next, self.x_mask_next, 
                                                         self.x_precomputed_features, self.y_data, 
                                                         self.y_data_prev, self.x_max_length],
                                           outputs=[self.softmax_cost_acc, self.softmax_cost, self.training_misclassification_first_acc, self.training_misclassification_second_acc, self.training_preds_first, self.training_preds_second], 
                                           on_unused_input='warn', name="eval_fn")
        return self.eval_fn

    def build_next_probs_function(self):
        if not hasattr(self, 'next_probs_fn'):
            outputs, hd = self.utterance_decoder.build_next_probs_predictor(self.beam_hs, self.beam_source, prev_state=self.beam_hd)
            self.next_probs_fn = theano.function(inputs=[self.beam_hs, self.beam_source, self.beam_hd],
                outputs=[outputs, hd],
                name="next_probs_fn")
        return self.next_probs_fn

    def __init__(self, state):
        Model.__init__(self)

        # Ensure backwards compatability by setting undefined configuration flags to their default values
        if not 'bidirectional_encoder' in state:
            state['bidirectional_encoder'] = False

        if not 'multiplicative_input_from_encoders' in state:
            state['multiplicative_input_from_encoders'] = False

        if not 'use_precomputed_features' in state:
            state['use_precomputed_features'] = False

        if not 'mlp_activation_function' in state:
            state['mlp_activation_function'] = 'rectifier'

        if not 'load_pretrained_rnns' in state:
            state['load_pretrained_rnns'] = False

        if not 'use_rnn_features' in state:
            state['use_rnn_features'] = True

        if not 'condition_on_previous_speaker_class' in state:
            state['condition_on_previous_speaker_class'] = False

        self.state = state
        self.global_params = []

        self.__dict__.update(state)
        self.rng = numpy.random.RandomState(state['seed']) 

        # Load dictionary
        raw_dict = cPickle.load(open(self.dictionary, 'r'))

        # Dictionaries to convert str to idx and vice-versa
        self.str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in raw_dict])
        self.idx_to_str = dict([(tok_id, tok) for tok, tok_id, freq, _ in raw_dict])

        # Backwards compatibility to older dictionaries
        if '<pause>' in self.idx_to_str:
            assert(self.idx_to_str['<pause>'] == self.pause_sym) # If this fail, set pause_sym appropriately.

        # Extract document (dialogue) frequency for each word
        self.word_freq = dict([(tok_id, freq) for _, tok_id, freq, _ in raw_dict])
        self.document_freq = dict([(tok_id, df) for _, tok_id, _, df in raw_dict])

        if '</s>' not in self.str_to_idx:
           raise Exception("Error, malformed dictionary!")
         
        # Number of words in the dictionary 
        self.idim = len(self.str_to_idx)
        self.state['idim'] = self.idim
        logger.debug("idim: " + str(self.idim))

        logger.debug("Initializing Theano variables")
        self.x_data_prev = T.imatrix('x_data_prev')
        self.x_mask_prev = T.matrix('x_mask_prev')
        self.x_data_next = T.imatrix('x_data_next')
        self.x_mask_next = T.matrix('x_mask_next')
        self.x_precomputed_features = T.matrix('x_precomputed_features')

        self.y_data = T.imatrix('y_data')
        self.y_data_prev = T.imatrix('y_data_prev')

        self.x_max_length = T.iscalar('x_max_length')

        self.y_training = self.y_data
        self.y_training_prev = self.y_data_prev

        self.y_extra = T.neq(self.y_data[0], 0)*T.neq(self.y_data[0], 5).dimshuffle('x', 0)
        self.y_training = T.concatenate([self.y_extra, self.y_data]) # 


        # Build word embeddings, which are shared throughout the model
        if self.initialize_from_pretrained_word_embeddings == True:
            # Load pretrained word embeddings from pickled file
            logger.debug("Loading pretrained word embeddings")
            pretrained_embeddings = cPickle.load(open(self.pretrained_word_embeddings_file, 'r'))

            # Check all dimensions match from the pretrained embeddings
            assert(self.idim == pretrained_embeddings[0].shape[0])
            assert(self.rankdim == pretrained_embeddings[0].shape[1])
            assert(self.idim == pretrained_embeddings[1].shape[0])
            assert(self.rankdim == pretrained_embeddings[1].shape[1])

            self.W_emb_pretrained_mask = theano.shared(pretrained_embeddings[1].astype(numpy.float32), name='W_emb_mask')
            self.W_emb = add_to_params(self.global_params, theano.shared(value=pretrained_embeddings[0].astype(numpy.float32), name='W_emb'))
        else:
            # Initialize word embeddings randomly
            self.W_emb = add_to_params(self.global_params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='W_emb'))


        # Initialize and build prev-encoder
        logger.debug("Initializing prev-encoder")
        self.encoder_prev = TextEncoder(self.state, self.rng, self.W_emb, self, 'prev')

        logger.debug("Build prev-encoder")
        self.h_prev = self.encoder_prev.build_encoder(self.x_data_prev, self.x_mask_prev)

        # Initialize and build next-encoder
        logger.debug("Initializing next-encoder")
        self.encoder_next = TextEncoder(self.state, self.rng, self.W_emb, self, 'next')

        logger.debug("Build next-encoder")
        self.h_next = self.encoder_next.build_encoder(self.x_data_next, self.x_mask_next)

        # If the encoder RNNs are bidirectional, we need to create the backward running RNNs
        if self.bidirectional_encoder:
            self.x_data_prev_reversed = self.x_data_prev[::-1]
            self.x_mask_prev_reversed = self.x_mask_prev[::-1]
            self.x_data_next_reversed = self.x_data_next[::-1]
            self.x_mask_next_reversed = self.x_mask_next[::-1]

            # Initialize and build prev-encoder reversed
            logger.debug("Initializing prev-encoder reversed")
            self.encoder_prev_reversed = TextEncoder(self.state, self.rng, self.W_emb, self, 'prev_reversed')

            logger.debug("Build prev-encoder reversed")
            self.h_prev_reversed = self.encoder_prev.build_encoder(self.x_data_prev_reversed, self.x_mask_prev_reversed)

            # Initialize and build next-encoder reversed
            logger.debug("Initializing next-encoder reversed")
            self.encoder_next_reversed = TextEncoder(self.state, self.rng, self.W_emb, self, 'next_reversed')

            logger.debug("Build next-encoder reversed")
            self.h_next_reversed = self.encoder_next.build_encoder(self.x_data_next_reversed, self.x_mask_next_reversed)
            if self.state['multiplicative_input_from_encoders']:
                self.decoder_input = T.concatenate([self.h_prev[-1] * self.h_next[-1], self.h_prev_reversed[-1] * self.h_next_reversed[-1]], axis=1)
            else:
                self.decoder_input = T.concatenate([self.h_prev[-1], self.h_next[-1], self.h_prev_reversed[-1], self.h_next_reversed[-1]], axis=1)
        else:
            if self.state['multiplicative_input_from_encoders']:
                self.decoder_input = self.h_prev[-1] * self.h_next[-1]
            else:
                self.decoder_input = T.concatenate([self.h_prev[-1], self.h_next[-1]], axis=1)

        # Trick to enable logistic regression on precomputed features
        if not self.use_rnn_features:
            self.decoder_input = 0*self.decoder_input

        # If using precomputed features append them as input to the decoder
        if self.use_precomputed_features:
            self.decoder_final_input = T.concatenate([self.decoder_input, self.x_precomputed_features.dimshuffle(1, 0)], axis=1)
        else:
            self.decoder_final_input = self.decoder_input

        logger.debug("Build decoder (EVAL)")
        self.decoder = TextDecoder(self.state, self.rng, self)

        self.full_probs, self.target_probs = self.decoder.build_decoder(self.decoder_final_input, self.y_training, self.y_training_prev, mode=TextDecoder.EVALUATION)

        # Prediction cost
        self.softmax_cost = -T.log(self.target_probs)
        self.softmax_cost_acc = T.sum(self.softmax_cost)

        # Compute training cost, which equals standard cross-entropy error
        self.training_cost = self.softmax_cost_acc

        # Prediction accuracy
        self.training_preds = T.argmax(self.full_probs, axis=1)
        # Compute speaker class prediction
        self.training_preds_first = self.training_preds[0:self.y_training[0].shape[0]]
        # Compute turn taking or non turn-taking class prediction
        self.training_preds_second = self.training_preds[self.y_training[0].shape[0]:2*self.y_training[0].shape[0]]

        # Compute speaker class and turn taking misclassification errors
        self.training_misclassification_first = T.neq(self.training_preds_first, self.y_training[0]).flatten()
        self.training_misclassification_second = T.neq(self.training_preds_second, self.y_training[1]).flatten()

        # Compute accumulated misclassification errors
        self.training_misclassification_first_acc = T.sum(self.training_misclassification_first)
        self.training_misclassification_second_acc = T.sum(self.training_misclassification_second)

        # Add params to list
        self.params = self.global_params + self.encoder_prev.params + self.encoder_next.params + self.decoder.params
        assert len(set(self.params)) == (len(self.global_params) + len(self.encoder_prev.params) + len(self.encoder_next.params) + len(self.decoder.params))

        # If the model is bidirectional, add the backward RNNs parameters to the list too
        if self.bidirectional_encoder:
            self.params += self.encoder_prev_reversed.params + self.encoder_next_reversed.params

        # Add gradient descent updates to Theano training function
        self.updates = self.compute_updates(self.training_cost, self.params)




