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

    start = time()
    fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(REPO_HOME+'data/wiki-news-300d-1M.vec', limit=MAX_VOCAB)
    print 'FastText loading: {:.2f}sec'.format(time()-start)

    # Load PronunciationDictionary' constructed by augmenting the CMUdict phonetic dictionary (nltk.corpus.cmudict.dict())
    start = time()
    grapheme_to_word_dict = PronunciationDictionary.load(REPO_HOME+'data/pronunciation_dictionary.pkl')
    print 'PronunciationDictionary loading: {:.2f}sec'.format(time()-start)    

    start = time()
    subword_frequency = SubwordFrequency.load(REPO_HOME+'data/subword_frequency.pkl')
    print 'SubwordFrequency loading: {:.2f}sec'.format(time()-start)    
    
    # Instead of typical input loop, run over first 200 random word pairs, and dump results to a file
    infile_name = '/Users/jonathansimon/code/what-do-you-call-a-bot/data/random_word_pairs.txt'
    with open(infile_name) as infile:
        pairs = [line.strip().split() for line in infile.readlines()]
        pairs = pairs[:200] # only care about top 200

    outfile_name = '/Users/jonathansimon/code/what-do-you-call-a-bot/data/entendrepeneur_top3_results.txt'
    with open(outfile_name, 'w') as outfile:
        for i, (grapheme1,grapheme2) in enumerate(pairs):

            # Gross that I have to pass fasttext_model as an input, but whatever...
            nearest_graphemes1 = get_semantic_neighbor_graphemes(grapheme1, fasttext_model)
            nearest_graphemes2 = get_semantic_neighbor_graphemes(grapheme2, fasttext_model)

            nearest_words1 = [grapheme_to_word_dict.get_word(grapheme) for grapheme in nearest_graphemes1 if grapheme_to_word_dict.get_word(grapheme)]
            nearest_words2 = [grapheme_to_word_dict.get_word(grapheme) for grapheme in nearest_graphemes2 if grapheme_to_word_dict.get_word(grapheme)]

            portmanteaus = get_portmanteaus(nearest_words1, nearest_words2, subword_frequency)

            # write input puts to head of line
            outfile.write('{},{}'.format(grapheme1,grapheme2))

            # if fewer than 3 are produced... replace remainer with "NULL"
            for j in range(3):
                if len(portmanteaus) > j:
                    outfile.write(',{}'.format(portmanteaus[j].grapheme_portmanteau1))
                else:
                    outfile.write(',NULL')
            
            # write newline break
            outfile.write('\n')

            print 'Finished generating {} rows, only {} remaining'.format(i+1, 200-i-1)