import cPickle
import numpy
import collections
import nltk
from nltk.corpus import sentiwordnet as swn

# Precomputed means and standard deviations for NLP features. 
# These will be used to normalize the NLP features to have mean zero and standard deviation one.

mean_feature_vector = numpy.asarray([2.96562195e-01, 4.03184026e-01, 0.00000000e+00, 2.83597261e-01, 3.86272103e-01, 0.00000000e+00, 1.29648941e-02, 1.69119667e-02, 0.00000000e+00, 3.83523852e-01, 4.88516361e-01, 0.00000000e+00, 3.57355364e-02, 3.31480950e-02, 2.58744159e-03, 6.03232346e-02, 3.00527946e-03, 4.07674350e-02, 1.89743135e-02, 1.39636518e-02, 9.12766084e-02, 2.82696541e-02, 5.77109912e-03, 1.80329889e-01, 7.74212405e-02, 4.15588282e-02, 1.84313469e-02, 2.89334208e-01, 8.03639814e-02, 7.08839949e-03, 1.03005216e-01, 1.25590824e-02, 5.49459979e-02, 2.03205943e-02, 1.48656536e-02, 7.99095854e-02, 3.00323889e-02, 1.82071198e-02, 1.72992900e-01, 8.75562578e-02, 5.18690273e-02, 1.59278195e-02, 2.56350726e-01, 5.83456308e-02, 6.99547306e-03, 1.18468814e-01, -9.55456588e-03, -1.41476160e-02, -1.34221860e-03, -9.00521118e-04, 1.14214439e-02, -1.77389185e-03, -1.24440547e-02, 7.25587271e-03, -1.02028111e-02, -1.02901896e-02, 2.50028004e-03, 3.28096189e-02, 2.20465120e-02, 8.77619241e-05, -1.54667506e-02, 1.41656613e-02, 7.88936093e-02, 2.99874451e-02, 2.12431140e-02, 1.38636976e-01, 4.97212298e-02, 2.16563269e-02, 2.37119108e-01, 1.24050483e-01, 7.09664896e-02, 2.68747602e-02, 2.96336830e-01, 9.93782431e-02, 1.09612867e-02, 1.70919865e-01, 6.95624161e+00, 6.50517416e+00, 4.46982265e-01, 1.22936440e+00], dtype='float32')

std_feature_vector =  numpy.asarray([0.49554962, 0.61514539, 0.01, 0.49025708, 0.61415344, 0.01, 0.63884366, 0.78571159, 0.01, 0.51067924, 0.61547154, 0.01, 0.21814735, 0.20856784, 0.28316221, 0.2770144, 0.01373707, 0.12397484, 0.0552241, 0.03703553, 0.22331558, 0.0944486, 0.0462641, 0.29226318, 0.17671695, 0.10085059, 0.05579286, 0.31407589, 0.14722739, 0.02853682, 0.2448494, 0.03326115, 0.12569752, 0.05854797, 0.03541771, 0.1909824, 0.07585674, 0.0600559, 0.28556755, 0.16632196, 0.11589559, 0.04712084, 0.2748251, 0.12178376, 0.0249551, 0.25520995, 0.03505171, 0.17603439, 0.07924691, 0.05013117, 0.28841811, 0.11690302, 0.07141439, 0.3811253, 0.2390068, 0.15077953, 0.07136076, 0.39477608, 0.18730916, 0.03574936, 0.33551174, 0.03352964, 0.15809451, 0.07342678, 0.04543759, 0.2533873, 0.10578044, 0.06910371, 0.29847983, 0.20470271, 0.13351575, 0.06620995, 0.26295692, 0.16042583, 0.03404612, 0.28902417, 1.0670743, 2.14937806, 2.31377745, 2.00574207], dtype='float32')

class NLPTools():
    def __init__(self, state):
        # Load dictionary
        self.raw_dict = cPickle.load(open(state['dictionary'], 'r'))
        # There are 614 movie scripts, but we add one to inverse-document frequency calculations
        self.document_count = 614 + 1 

        # Dictionaries to convert str to idx and vice-versa
        self.str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in self.raw_dict])
        self.idx_to_str = dict([(tok_id, tok) for tok, tok_id, freq, _ in self.raw_dict])

        self.idx_to_polarity = dict()
        self.document_freq = dict([(tok_id, df) for _, tok_id, _, df in self.raw_dict])

        for key, idx in self.str_to_idx.iteritems():
            r = numpy.zeros((3))
            l = list(swn.senti_synsets(key))
            if l and len(l) > 0:
                r[0] = l[0].neg_score()
                r[1] = l[0].obj_score()
                r[1] = l[0].pos_score()

            self.idx_to_polarity[idx] = r

        # Build list of swear words and bad words
        # Taken on July 26, 2015 from https://github.com/shutterstock
        # /List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en
        bad_words = open('bad_words_list.txt', 'r').readlines()
        bad_words_set = set()
        for word in bad_words:
            bad_words_set.add(word.strip())


        self.idx_to_bad_word = dict()
        for key, idx in self.str_to_idx.iteritems():
            if key in bad_words_set:
                self.idx_to_bad_word[idx] = 1.0
            else:
                self.idx_to_bad_word[idx] = 0.0

        # Build dialogue act tagger
        # Based on code here: http://www.nltk.org/book/ch06.html
        dlg_tagger_posts = nltk.corpus.nps_chat.xml_posts()[:10000]
        dlg_tagger_featuresets = [(self.dialogue_act_features(post.text), post.get('class'))
                       for post in dlg_tagger_posts]
        dlg_tagger_size = int(len(dlg_tagger_featuresets) * 0.1)
        dlg_tagger_train_set, dlg_tagger_test_set = dlg_tagger_featuresets[dlg_tagger_size:], dlg_tagger_featuresets[:dlg_tagger_size]
        self.dlg_tagger_classifier = nltk.NaiveBayesClassifier.train(dlg_tagger_train_set)

        # Print accuracy on test set
        #print(nltk.classify.accuracy(self.dlg_tagger_classifier, dlg_tagger_test_set))

    def dialogue_act_features(self, post):
        features = {}
        for word in nltk.word_tokenize(post):
            features['contains({})'.format(word.lower())] = True
        return features

    def get_features(self, prev_sentence, next_sentence, token_counter):
        # Get negative, objective and positive sentiment polarities
        prev_scores = numpy.zeros((3), dtype='float32')
        for i in range(len(prev_sentence)):
            prev_scores += self.idx_to_polarity[prev_sentence[i]]

        next_scores = numpy.zeros((3), dtype='float32')
        for i in range(len(next_sentence)):
            next_scores += self.idx_to_polarity[next_sentence[i]]

        # Count number of bad (e.g. swear) words     
        prev_bad_words = numpy.zeros((1), dtype='float32')
        for i in range(len(prev_sentence)):  
            prev_bad_words += self.idx_to_bad_word[prev_sentence[i]]

        next_bad_words = numpy.zeros((1), dtype='float32')
        for i in range(len(next_sentence)):  
            next_bad_words += self.idx_to_bad_word[next_sentence[i]]

        # Compute probabilities for all dialogue act types of each sentence
        prev_sentence_string = ''
        for i in range(len(prev_sentence)-1):
            prev_sentence_string += self.idx_to_str[prev_sentence[i]] + ' '
        if len(prev_sentence) > 0:
            prev_sentence_string += self.idx_to_str[prev_sentence[-1]]

        next_sentence_string = ''
        for i in range(len(next_sentence)-1):
            next_sentence_string += self.idx_to_str[next_sentence[i]] + ' '
        if len(next_sentence) > 0:
            next_sentence_string += self.idx_to_str[next_sentence[-1]]


        prev_dialog_act_classes = self.dlg_tagger_classifier.prob_classify(self.dialogue_act_features(prev_sentence_string))
        next_dialog_act_classes = self.dlg_tagger_classifier.prob_classify(self.dialogue_act_features(next_sentence_string))

        prev_probabilities = numpy.zeros((len(self.dlg_tagger_classifier._labels)), dtype='float32')
        next_probabilities = numpy.zeros((len(self.dlg_tagger_classifier._labels)), dtype='float32')


        for label_index in range(len(self.dlg_tagger_classifier._labels)):
            label = self.dlg_tagger_classifier._labels[label_index]
            prev_probabilities[label_index] = prev_dialog_act_classes.prob(label)
            next_probabilities[label_index] = next_dialog_act_classes.prob(label)

        # Finally, compute TF-IDF frequencies for both sentences (simply as sum of TF-IDF over the words)
        prev_token_counter = collections.Counter(prev_sentence)
        prev_tf_idf = numpy.zeros((1), dtype='float32')
        for word_index in prev_token_counter.keys():
            prev_tf_idf += prev_token_counter[word_index] * numpy.log(self.document_count/max(1, self.document_freq[word_index]))
        if len(prev_token_counter.keys()) > 0:
            prev_tf_idf /= len(prev_token_counter.keys())

        next_token_counter = collections.Counter(next_sentence)
        next_tf_idf = numpy.zeros((1), dtype='float32')
        for word_index in next_token_counter.keys():
            next_tf_idf += next_token_counter[word_index] * numpy.log(self.document_count/max(1, self.document_freq[word_index]))
        if len(next_token_counter.keys()) > 0:
            next_tf_idf /= len(next_token_counter.keys())

        # Return entire vector of concatenated features; normalized to have mean zero and standard deviation one.
        # 3 + 3 + 3 + 3 + 1 + 1 + 1 + 1 + 15 + 15 + 15 + 15 + 1 + 1 + 1 + 1
        r = numpy.concatenate([prev_scores, next_scores, (prev_scores - next_scores), numpy.abs(prev_scores - next_scores), prev_bad_words, next_bad_words, (prev_bad_words-next_bad_words), numpy.abs(prev_bad_words - next_bad_words), prev_probabilities, next_probabilities, (prev_probabilities-next_probabilities), numpy.abs(prev_probabilities-next_probabilities), prev_tf_idf, next_tf_idf, prev_tf_idf - next_tf_idf, numpy.abs(prev_tf_idf-next_tf_idf)], axis=0).astype('float32')

        return (r - mean_feature_vector)/std_feature_vector
