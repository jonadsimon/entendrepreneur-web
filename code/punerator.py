import sys
import gensim
import nltk
import numpy as np
from global_constants import *
from helper_utils import *
from pronunciation_dictionary import PronunciationDictionary
import pdb

# Steps:
# 0. Accept inputs
# 1. Map to nearest neighbors (catch errors/make suggestions)
# 2. Map to phonemes/words
# 3. n^2 search to identify portmanteaus and rhyms
# 4. print results

if __name__ == '__main__':

    options = parse_options(sys.argv)

    if not options['test']:
        # Load Google's pre-trained Word2Vec model
        if not options['fast']:
            word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(REPO_HOME+'data/GoogleNews-vectors-negative300.bin.gz', binary=True)  
        else:
            # restrict to a small subset of the overall vector set
            word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(REPO_HOME+'data/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=FAST_LIMIT)

    # Load PronunciationDictionary' constructed by augmenting the CMUdict phonetic dictionary (nltk.corpus.cmudict.dict())
    grapheme_to_word_dict = PronunciationDictionary.load(REPO_HOME+'data/pronunciation_dictionary.pkl')

    while True:
        if not options['test']:
            input_string = raw_input("\nSeed words:  ")
            status, message = validate_input(input_string, word2vec_model.vocab)
            # If anything is wrong with the input, set the status to 1, print a message describing the problem, and skip the rest of the logic
            if status == 1:
                print message
                continue
            
            grapheme1, grapheme2 = parse_input(input_string)

            # Gross that I have to pass word2vec_model as an input, but whatever...
            nearest_graphemes1 = get_semantic_neighbor_graphemes(grapheme1, word2vec_model)
            nearest_graphemes2 = get_semantic_neighbor_graphemes(grapheme2, word2vec_model)

            nearest_words1 = [grapheme_to_word_dict.get_word(grapheme) for grapheme in nearest_graphemes1 if grapheme_to_word_dict.get_word(grapheme)]
            nearest_words2 = [grapheme_to_word_dict.get_word(grapheme) for grapheme in nearest_graphemes2 if grapheme_to_word_dict.get_word(grapheme)]

        else:
            grapheme1, grapheme2 = parse_input(TEST_INPUT)
            # !!! NEED TO SPECIAL-CASE THIS SITUATION, OR ELSE REDEFINE "TEST" TO MEAN "FAST"
            nearest_words1 = [grapheme_to_word_dict.get_word(grapheme1)]
            nearest_words2 = [grapheme_to_word_dict.get_word(grapheme2)]

        portmanteaus = get_portmanteaus(nearest_words1, nearest_words2, grapheme_to_word_dict)
        # pdb.set_trace()
        # rhymes = get_rhymes(nearest_words1, nearest_words2)

        for i, portmanteau in enumerate(portmanteaus):
            if i >= MAX_PORTMANTEAUS:
                break
            if options['debug']:
                print repr(portmanteau)
            else:
                print portmanteau

        # for i, rhyme in enumerate(rhymes):
        #     if i >= MAX_RHYMES:
        #         break
        #     print rhyme

        # if it's a test run, we only want to run the while-loop once
        if options['test']:
            break