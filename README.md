# dlg-segmenter

Dialog speaker and turn taking classification model based on GRU recurrent neural networks (RNNs).

The model learns to predict the speaker segmentation from dialogue: 
- speaker changes and speaker identity: <first_speaker>, <second_speaker>, <third_speaker> and <minor_speaker>,
          where it is assumed that the first three speakers are sorted by frequency (e.g. so that <first_speaker> is
          the unique speaker with the most utterances), and that <minor_speaker> covers all other speakers.
- auxiliary segmentation specific to movies: <voice_over> and <off_screen>. These can safely be for non-movie data.

The input to the model is expected to be two sequences of tokens (words), one sequence before the segmentation label and one sequence after the segmentation label. Each sequence is processed by a separate GRU RNN. The final RNN hidden states are then multiplied together elementwise, linearly projected to a subspace and given as input to a single-layer MLP function. Additional NLP features may be given as input to the MLP as well. Finally, the MLP output is transformed with a softmax function to give probabilities over speaker and auxiliary segmentation.

The "bad_words_list.txt" was copied from github.com/shutterstock/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en.
