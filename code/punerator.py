import sys
import gensim
import nltk
import numpy as np
from global_constants import *
from helper_utils import *
from pronunciation_dictionary import PronunciationDictionary
from subword_frequency import SubwordFrequency
from time import time

if __name__ == '__main__':

    options = parse_options(sys.argv)

    start = time()
    if not options['test']:
        # Load Facebook's pre-trained FastVec model
        fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(REPO_HOME+'data/word_vectors/wiki-news-300d-1M.vec', limit=MAX_VOCAB)
            
    print 'FastText loading: {:.2f}sec'.format(time()-start)

    # Load PronunciationDictionary' constructed by augmenting the CMUdict phonetic dictionary (nltk.corpus.cmudict.dict())
    start = time()
    grapheme_to_word_dict = PronunciationDictionary.load(REPO_HOME+'data/pronunciation_dictionary.pkl')
    print 'PronunciationDictionary loading: {:.2f}sec'.format(time()-start)    

    start = time()
    subword_frequency = SubwordFrequency.load(REPO_HOME+'data/subword_frequency.pkl')
    print 'SubwordFrequency loading: {:.2f}sec'.format(time()-start)    
    

    while True:
        if not options['test']:
            input_string = raw_input("\nSeed words:  ")
            status, message = validate_input(input_string, fasttext_model.vocab)
            # If anything is wrong with the input, set the status to 1, print a message describing the problem, and skip the rest of the logic
            if status == 1:
                print message
                continue
            
            grapheme1, grapheme2 = parse_input(input_string)

            # Gross that I have to pass fasttext_model as an input, but whatever...
            start = time()
            nearest_graphemes1 = get_semantic_neighbor_graphemes(grapheme1, fasttext_model)
            nearest_graphemes2 = get_semantic_neighbor_graphemes(grapheme2, fasttext_model)
            print 'Finding nearest-neighbor graphemes: {:.2f}sec'.format(time()-start)

            start = time()
            nearest_words1 = [grapheme_to_word_dict.get_word(grapheme) for grapheme in nearest_graphemes1 if grapheme_to_word_dict.get_word(grapheme)]
            nearest_words2 = [grapheme_to_word_dict.get_word(grapheme) for grapheme in nearest_graphemes2 if grapheme_to_word_dict.get_word(grapheme)]
            print 'Nearest-neighbor graphemes to words: {:.2f}sec'.format(time()-start)    

        else:
            grapheme1, grapheme2 = parse_input(TEST_INPUT)
            nearest_words1 = [grapheme_to_word_dict.get_word(grapheme1)]
            nearest_words2 = [grapheme_to_word_dict.get_word(grapheme2)]

        start = time()
        portmanteaus = get_portmanteaus(nearest_words1, nearest_words2, subword_frequency)
        print 'Portmanteau generation: {:.2f}sec'.format(time()-start)

        start = time()
        rhymes = get_rhymes(nearest_words1, nearest_words2, subword_frequency)
        print 'Rhyme generation: {:.2f}sec'.format(time()-start)

        print '''
        ########################
        ##### PORTMANTEAUS #####
        ########################
        '''
        for i, portmanteau in enumerate(portmanteaus):
            if i >= MAX_PORTMANTEAUS:
                break
            if options['debug']:
                print repr(portmanteau)
            else:
                print portmanteau

        print '''
        ##################
        ##### RHYMES #####
        ##################
        '''
        for i, rhyme in enumerate(rhymes):
            if i >= MAX_RHYMES:
                break
            if options['debug']:
                print repr(rhyme)
            else:
                print rhyme

        # if it's a test run, we only want to run the while-loop once
        if options['test']:
            break