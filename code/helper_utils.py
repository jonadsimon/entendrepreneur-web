from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from portmanteau import Portmanteau
from rhyme import Rhyme
from global_constants import MAX_NEIGHBORS, NEAR_MISS_VOWELS, NEAR_MISS_CONSONANTS
import io

def parse_options(args):
    '''
    Parse and return the program option provided by the user at runtime.
    All provided option arguments are preceded by the '--' string.
    Currently the options supported are:
        * --test : runs the script once with inputs provided by the TEST_INPUT string in global_constants.py
        * --debug : prints additional information about each Portmanteau and Rhyme when generating output
    '''
    options = defaultdict(bool)
    for arg in args:
        if arg[:2] == '--':
           options[arg[2:]] = True
    return options

def alternative_grapheme_capitalizations(grapheme):
    '''
    For a given grapheme, returns a list containing:
    1) the grapheme as-is
    2) the grapheme all-lower-cased
    3) the grapheme all-lowercased except for the first letter, which is capitalized
    '''
    capitalization_alternatives = [grapheme]
    capitalization_alternatives.append(grapheme.lower())
    if len(grapheme) == 1:
        capitalization_alternatives.append(grapheme[0].upper())
    elif len(grapheme) > 1:
        capitalization_alternatives.append(grapheme[0].upper()+grapheme[1:].lower())
    return capitalization_alternatives

def validate_input(input_string, recognized_graphemes):
    '''
    Verify that the user-provided input string is comprised of two graphemes, and that both graphemes are present in FastText
    '''
    if type(input_string) is not str or len(input_string.split()) != 2:
        status = 1
        message = 'Error: Input should be of the form "[word1] [word2]"'
    else:
        grapheme1, grapheme2 = input_string.split()

        # Each grapheme itself or one of its "alternative_grapheme_capitalizations" must exists in the FastText corpus
        grapheme1_exists = any([g in recognized_graphemes for g in alternative_grapheme_capitalizations(grapheme1)])
        grapheme2_exists = any([g in recognized_graphemes for g in alternative_grapheme_capitalizations(grapheme2)])

        # Return the appropriate error if either of the graphemes cannot does not exist in FastText, else return status=0
        if not grapheme1_exists and grapheme2_exists:
            status = 1
            message = "Error: '{}' is not a recognized word, please check the spelling".format(grapheme1)
        elif grapheme1_exists and not grapheme2_exists:
            status = 1
            message = "Error: '{}' is not a recognized word, please check the spelling".format(grapheme2)
        elif not grapheme1_exists and not grapheme2_exists:
            status = 1
            message = "Error: neither '{}' nor '{}' is a recognized word, please check the spelling".format(grapheme1, grapheme2)
        else:
            status = 0
            message = ''

    return status, message

def get_fasttext_neighbors(grapheme, fasttext_model):
    '''
    For a given grapheme, return the MAX_NEIGHBORS many nearest neighbor graphemes to it
    (or to one of its alternative_grapheme_capitalizations) in the FastText embedding
    '''
    for g in alternative_grapheme_capitalizations(grapheme):
        if g in fasttext_model.vocab:
            return list(zip(*fasttext_model.most_similar(positive=[g], topn=MAX_NEIGHBORS))[0])
    else:
        raise 'This code path should NEVER execute, something has gone HORRIBLY wrong'

def get_shortest_lemma(grapheme, lemmatizer=WordNetLemmatizer(), stemmer=PorterStemmer()):
    '''
    Don't want to produce multiple portmanteaus/rhymes where one of the words differs only by its e.g. pluralization
    Therefore for each grapheme, identify its shortest lemma or stem (assuming that the stem is a valid word)

    Because WordNet classifies words by part-of-speech, we need to check check all possible POS's to make sure that
    the *shortest* lemma has been found. If we manually check all parts of speech, it defaults to using the lemmas
    associated with the first synset, which may not be what we want.

    The parts of speech recognized by WordNet are:
      n : NOUN
      v : VERB
      a : ADJECTIVE
      s : ADJECTIVE SATELLITE
      r : ADVERB

    Note: For efficiency, we instantiate the lemmatizer in advance, so that it doesn't need to be reinstantiated on each run
    '''

    shortest_lemma = grapheme

    # First attempt to trim using lemmatizer
    # Be sure to check all parts-of-speech for lemmas
    for pos in ['n','v','a','s','r']:
        lemma = lemmatizer.lemmatize(grapheme, pos)
        if len(lemma) < len(shortest_lemma):
            shortest_lemma = lemma

    # Next attempt to trim using stemmer; further trim down whatever was produced by the lemmatizer
    # Stems are not always valid words, so verify that the result is present in WordNet
    stem = stemmer.stem(shortest_lemma)
    if len(wn.synsets(stem)) >= 1 and len(stem) < len(shortest_lemma):
        shortest_lemma = stem

    return shortest_lemma

def get_semantic_neighbor_graphemes(grapheme, fasttext_model):
    '''
    For a given input grapheme, finds its FastText nearest neighbors, and then cleans these neighbor graphemes
    by mapping each to its shortest variant using 'get_shortest_lemma', and deduplicating the results
    '''
    # Not ideal that 'fasttext_model' has to be passed twice... consider refactoring
    fasttext_neighbors = get_fasttext_neighbors(grapheme, fasttext_model)

    # FastText sometimes returns funky unicode characters like umlouts, so make sure to catch/discard these before continuing
    # Make sure to include the input word itself as one of the neighbors
    fasttext_neighbors_clean = [grapheme]
    for g in fasttext_neighbors:
        try:
            fasttext_neighbors_clean.append(str(g))
        except:
            pass

    # For each neighbor grapheme, perform the following operations:
    # 1) downcase
    # 2) find shortest lemma or valid stem
    # 3) deduplicate
    wnl = WordNetLemmatizer()
    semantic_neighbor_graphemes = map(lambda g: get_shortest_lemma(g.lower(), wnl), fasttext_neighbors_clean)
    semantic_neighbor_graphemes = set(semantic_neighbor_graphemes)

    return semantic_neighbor_graphemes

def get_portmanteaus(words1_neighbors, words2_neighbors, subword_frequency):
    '''
    Given a two lists of words, attempt to construct portmanteaus out of each of the
    |words1_neighbors| x |words2_neighbors| many word pairs. Order the generated portmanteaus
    by quality before returning.
    '''
    # Store the results using a set to avoid duplicates in case the same words appears in both sets
    portmanteau_set = set()
    for neighbor1 in words1_neighbors:
        for neighbor2 in words2_neighbors:
            # If the words are identical, as sometimes happens, skip this word pair
            if neighbor1.grapheme == neighbor2.grapheme:
                continue
            # Generate forward-ordered portmanteau
            # Not ideal that 'subword_frequency' has to be passed twice... consider refactoring
            portmanteau, status, message = Portmanteau.get_pun(neighbor1, neighbor2, subword_frequency)
            if status == 0:
                portmanteau_set.add(portmanteau)
            # Generate reverse-ordered portmanteau
            # Not ideal that 'subword_frequency' has to be passed twice... consider refactoring
            portmanteau, status, message = Portmanteau.get_pun(neighbor2, neighbor1, subword_frequency)
            if status == 0:
                portmanteau_set.add(portmanteau)

    # Order the results in terms of portmanteau quality, with better portmanteaus appearing earlier
    portmanteau_list = list(portmanteau_set)
    portmanteau_list.sort(key=lambda x: x.ordering_criterion())

    return portmanteau_list

def get_rhymes(words1_neighbors, words2_neighbors, subword_frequency):
    '''
    Given a two lists of words, attempt to construct rhyme out of each of the
    |words1_neighbors| x |words2_neighbors| many word pairs. Order the generated rhymes
    by quality before returning.
    '''
    # Store the results using a set to avoid duplicates in case the same words appears in both sets
    rhyme_set = set()
    for neighbor1 in words1_neighbors:
        for neighbor2 in words2_neighbors:
            # If the words are identical, as sometimes happens, skip this word pair
            if neighbor1.grapheme == neighbor2.grapheme:
                continue
            # Generate the rhyme for only a single ordering, if the words need to be flipped
            # for quality reasons, that's handled within the 'get_rhyme' function
            # Not ideal that 'subword_frequency' has to be passed twice... consider refactoring
            rhyme, status, message = Rhyme.get_pun(neighbor1, neighbor2, subword_frequency)
            if status == 0:
                rhyme_set.add(rhyme)

    # Order the results in terms of rhyme quality, with better rhymes appearing earlier
    rhyme_list = list(rhyme_set)
    rhyme_list.sort(key=lambda x: x.ordering_criterion())

    return rhyme_list
