from nlp_tools import NLPTools 
from collections import OrderedDict

def prototype_state():
    state = {} 

    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['level'] = 'DEBUG'

    # String for unknown (out-of-vocabulary) tokens
    state['oov'] = '<unk>'

    # Number of tokens before and after the potential speaker segmentation token to condition on.
    state['max_length'] = 20

    # Special tokens need to be hardcoded, because model architecture may adapt depending on these
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = 2 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = 3 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = 4 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = 5 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = 6 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = 7 # voice over symbol <voice_over>
    state['off_screen_sym'] = 8 # off screen symbol <off_screen>
    state['pause_sym'] = 9 # pause sybmol <pause>, double check that this is correct for every new dictionary

    state['segmentation_token_count'] = 6

    # ----- ACTIVATION FUNCTION ---- 
    state['rec_activation'] = 'lambda x: T.tanh(x)'

    # Gating of encoder RNNs: either 'plain' (tanh activation function) or 'gated' (GRU activation function)
    state['sent_step_type'] = 'gated'

    # If on, will run four RNNs:
    # - One RNN forwards on the token sequence before the segmentation label
    # - One RNN backwards on the token sequence before the segmentation label
    # - One RNN forwards on the token sequence after the segmentation label
    # - One RNN backwards on the token sequence after the segmentation label
    state['bidirectional_encoder'] = False

    # Dimensionality of the hidden layer of MLP 
    # (which takes as input the hidden states of the two RNNs)
    state['mlp_out_dim'] = 50
    state['mlp_activation_function'] = 'linear' # 'rectifier', 'tanh' or 'linear'

    # If on, the RNN encoder hidden states will be multiplied by each other elementwise to give a new vector,
    # which is then given as input to the MLP.
    state['multiplicative_input_from_encoders'] = True

    # Dimensionality of hidden layers
    state['qdim'] = 512
    # Dimensionality of low-rank approximation
    state['rankdim'] = 256

    # Threshold to clip the gradient
    state['cutoff'] = 1
    # Learning rate for Adam
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 5
    state['cost_threshold'] = 1.003

    # Initialization configuration
    # Whether to initialize from word embeddings (e.g. Word2Vec embeddings)
    state['initialize_from_pretrained_word_embeddings'] = False
    # Word embeddings file (can be produced by running the "convert-wordemb-dict2emb-matrix" script) 
    state['pretrained_word_embeddings_file'] = ''
    # Whether or not to keep word embeddings fixed during training
    state['fix_pretrained_word_embeddings'] = False

    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'  

    # Batch size
    state['bs'] = 80

    # Sort by length groups of batches
    state['sort_k_batches'] = 20
   
    # Modify this in each specific prototype
    state['save_dir'] = './'
    
    # ----- TRAINING PROCESS -----
    # Frequency of training error reports (in number of batches)
    state['train_freq'] = 10
    # Validation frequency
    state['valid_freq'] = 5000
    # Number of batches to process
    state['loop_iters'] = 3000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1

    # ----- EVALUATION PROCESS -----
    # Signs used to denote end-of-sentence tokens
    state['sentence_break_signs'] = ['.', '!', '?', '</s>']

    # Use precomputed features
    state['use_precomputed_features'] = True
    state['precomputed_features_count'] = 80

    # Use RNN features (can be disabled to run logistic regression on precomputed features only)
    state['use_rnn_features'] = True

    # Condition softmax layer (logistic regression) on previous speaker class
    state['condition_on_previous_speaker_class'] = False

    return state

# This prototype is only for testing the model...
def prototype_test():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    state['test_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    state['valid_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    state['dictionary'] = "./tests/data/ttrain.dict.pkl"
    state['save_dir'] = "./tests/models/"

    # Handle pretrained word embeddings. Using this requires rankdim=10
    state['initialize_from_pretrained_word_embeddings'] = False
    state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl' 
    state['fix_pretrained_word_embeddings'] = False
    
    # Validation frequency
    state['valid_freq'] = 50
    
    state['prefix'] = "testmodel_" 
    state['updater'] = 'adam'

    state['sent_step_type'] = 'gated'

    state['bidirectional_encoder'] = False

    # If out of memory, modify this!
    state['bs'] = 10
    state['sort_k_batches'] = 1

    state['qdim'] = 20
    state['rankdim'] = 10

    state['nlp_tools'] = NLPTools(state)

    return state

def prototype_movies():
    state = prototype_state()
    
    # Fill your paths here! 
    state['train_dialogues'] = "Data/Training.dialogues.pkl"
    state['test_dialogues'] = "Data/Test.dialogues.pkl"
    state['valid_dialogues'] = "Data/Validation.dialogues.pkl"
    state['dictionary'] = "Data/Dataset.dict.pkl" 
    state['save_dir'] = "Output" 

    # Handle pretrained word embeddings.
    # These need to be recomputed if we want them for the 20K vocabulary.
    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = 'Data/Word2Vec_WordEmb_50Dim.pkl'
    state['fix_pretrained_word_embeddings'] = True
    
    # Validation frequency
    state['valid_freq'] = 1250
    
    state['prefix'] = "MovieScriptModel_" 
    state['updater'] = 'adam'
     
    # If out of memory, modify this!
    state['bs'] = 80

    state['qdim'] = 100
    state['rankdim'] = 50

    # Use precomputed features
    state['use_precomputed_features'] = True
    state['precomputed_features_count'] = 80
    state['nlp_tools'] = NLPTools(state)

    return state
