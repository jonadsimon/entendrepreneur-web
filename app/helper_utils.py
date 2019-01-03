from sqlalchemy import text
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from portmanteau import Portmanteau
from rhyme import Rhyme
from global_constants import MAX_NEIGHBORS
from app import db

def alternate_capitalizations(grapheme):
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

# TODO: refactor so this function is a classmethod of FasttextVector
def get_semantic_neighbor_graphemes(grapheme):
    '''
    Computes the cosine distance of the input grapheme's word vector with every other word vector in FasttextVectorElement,
    and returns the nearest MAX_NEIGHBORS many graphemes, along with the grapheme itself

    Very difficult to structure this query in SQLAlchemy's query meta-language,
    so just use 'execute' to run the raw SQL
    '''

    fv1_dot_fv2 = ' + '.join(["fv1.v{}*fv2.v{}".format(i+1,i+1) for i in range(150)])

    query = '''
    SELECT
        fv2.grapheme,
        ({}) cosine_similarity
    FROM fasttext_vectors fv1
    JOIN fasttext_vectors fv2
    ON fv1.grapheme = :grapheme
    ORDER BY 2 DESC
    LIMIT :max_neighbors + 1
    '''.format(fv1_dot_fv2)

    # Pass in the parameterized query params
    result = db.engine.execute(text(query), grapheme=grapheme, max_neighbors=MAX_NEIGHBORS)

    fasttext_neighbor_graphemes = [row['grapheme'] for row in result]

    # FastText sometimes returns funky unicode characters like umlouts, so make sure to catch/discard these before continuing
    fasttext_neighbor_graphemes_clean = []
    for g in fasttext_neighbor_graphemes:
        try:
            fasttext_neighbor_graphemes_clean.append(str(g))
        except:
            pass

    # For each neighbor grapheme, perform the following operations:
    # 1) downcase
    # 2) find shortest lemma or valid stem
    # 3) deduplicate
    semantic_neighbor_graphemes = map(lambda g: get_shortest_lemma(g.lower()), fasttext_neighbor_graphemes_clean)
    semantic_neighbor_graphemes = set(semantic_neighbor_graphemes)

    return semantic_neighbor_graphemes

def get_portmanteaus(words1_neighbors, words2_neighbors):
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
            portmanteau, status, message = Portmanteau.get_pun(neighbor1, neighbor2)
            if status == 0:
                portmanteau_set.add(portmanteau)
            # Generate reverse-ordered portmanteau
            portmanteau, status, message = Portmanteau.get_pun(neighbor2, neighbor1)
            if status == 0:
                portmanteau_set.add(portmanteau)

    # Order the results in terms of portmanteau quality, with better portmanteaus appearing earlier
    portmanteau_list = list(portmanteau_set)
    portmanteau_list.sort(key=lambda x: x.ordering_criterion())

    return portmanteau_list

def get_rhymes(words1_neighbors, words2_neighbors):
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
            rhyme, status, message = Rhyme.get_pun(neighbor1, neighbor2)
            if status == 0:
                rhyme_set.add(rhyme)

    # Order the results in terms of rhyme quality, with better rhymes appearing earlier
    rhyme_list = list(rhyme_set)
    rhyme_list.sort(key=lambda x: x.ordering_criterion())

    return rhyme_list
