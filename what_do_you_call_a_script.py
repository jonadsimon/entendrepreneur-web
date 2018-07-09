import sys
import gensim
import nltk
import numpy as np

########################
### GLOBAL CONSTANTS ###
########################

ARPABET_VOWELS = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY', 'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX'])
ARPABET_CONSONANTS = set(['B', 'CH', 'D', 'DH', 'DX', 'EL', 'EM', 'EN', 'F', 'G', 'H', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'NX', 'P', 'Q', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH'])

ARPABET_DIPHTHONGS = set(['AW', 'AY', 'EY', 'OW', 'OY'])
ARPABET_RHOTICS = set(['ER'])

MAX_PUNS = 100

ARPABET_PHONE_TO_PHONOLOGICAL_PHONE_DICT = {
    'AA': ['AA'],
    'AE': ['AE'],
    'AH': ['AH'],
    'AO': ['AO'],
    'AW': ['AA', 'UH'], # diphthong
    'AY': ['AA', 'IH'], # diphthong
    'B': ['B'],
    'CH': ['CH'],
    'D': ['D'],
    'DH': ['DH'],
    'EH': ['EH'],
    'ER': ['AO', 'R'], # rhotic; 'EH' instead of 'AO'?
    'EY': ['e', 'IH'], # diphthong
    'F': ['F'],
    'G': ['G'],
    'HH': ['HH'],
    'IH': ['IH'],
    'IY': ['IY'],
    'JH': ['JH'],
    'K': ['K'],
    'L': ['L'],
    'M': ['M'],
    'N': ['N'],
    'NG': ['NG'],
    'OW': ['o', 'UH'], # diphthong
    'OY': ['AO', 'IH'], # diphthong
    'P': ['P'],
    'R': ['R'],
    'S': ['S'],
    'SH': ['SH'],
    'T': ['T'],
    'TH': ['TH'],
    'UH': ['UH'],
    'UW': ['UW'],
    'V': ['V'],
    'W': ['W'],
    'Y': ['Y'],
    'Z': ['Z'],
    'ZH': ['ZH'],
}

PHONOLOGICAL_PHONE_TO_PHONOLOGICAL_FEATURE_DICT = {
    'AA': [-1,1,1,1,-1,-1,0,0,1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1], # variant
    'AE': [-1,1,1,1,-1,-1,0,0,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1],
    'AH': [-1,1,1,1,1,-1,0,0,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1], # treat as UH rather than AO
    'AO': [-1,1,1,1,1,-1,0,0,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1],
    'B': [1,-1,-1,1,-1,-1,0,0,-1,0,0,0,0,-1,0,1,-1,-1,-1,-1,-1,-1,-1],
    'CH': [1,-1,-1,-1,0,1,-1,1,1,1,-1,-1,-1,-1,0,-1,-1,-1,0,1,-1,1,-1],
    'D': [1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,-1,-1,-1,-1,-1],
    'DH': [1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,1,-1,-1,-1,-1],
    'EH': [-1,1,1,1,-1,-1,0,0,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1],
    'e': [-1,1,1,1,-1,-1,0,0,1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,-1],
    'F': [1,-1,-1,1,-1,-1,0,0,-1,0,0,0,0,-1,0,-1,-1,-1,1,1,-1,-1,-1],
    'G': [1,-1,-1,-1,0,-1,0,0,1,1,-1,1,-1,-1,0,1,-1,-1,-1,-1,-1,-1,-1],
    'HH': [-1,-1,-1,-1,0,-1,0,0,-1,0,0,0,0,-1,0,-1,1,-1,1,-1,-1,-1,-1],
    'IH': [-1,1,1,1,-1,-1,0,0,1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1],
    'IY': [-1,1,1,1,-1,-1,0,0,1,1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,-1],
    'JH': [1,-1,-1,-1,0,1,-1,1,1,1,-1,-1,-1,-1,0,1,-1,-1,0,1,-1,1,-1],
    'K': [1,-1,-1,-1,0,-1,0,0,1,1,-1,1,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1],
    'L': [1,1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,1,-1,1,-1,-1],
    'M': [1,1,-1,1,-1,-1,0,0,-1,0,0,0,0,-1,0,1,-1,-1,-1,-1,-1,-1,1],
    'N': [1,1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,-1,-1,-1,-1,1],
    'NG': [1,1,-1,-1,0,-1,0,0,1,1,-1,1,-1,-1,0,1,-1,-1,-1,-1,-1,-1,1],
    'o': [-1,1,1,1,1,-1,0,0,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1],
    'P': [1,-1,-1,1,-1,-1,0,0,-1,0,0,0,0,-1,0,-1,-1,-1,-1,-1,-1,-1,-1],
    'R': [1,1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,1,-1,-1,-1,-1],
    'S': [1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,-1,-1,-1,1,1,-1,-1,-1],
    'SH': [1,-1,-1,-1,0,1,-1,1,-1,0,0,0,0,-1,0,-1,-1,-1,1,-1,-1,-1,-1],
    'T': [1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,-1,-1,-1,-1,-1,-1,-1,-1],
    'TH': [1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,-1,-1,-1,1,-1,-1,-1,-1],
    'UH': [-1,1,1,1,1,-1,0,0,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1],
    'UW': [-1,1,1,1,1,-1,0,0,1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1],
    'V': [1,-1,-1,1,-1,-1,0,0,-1,0,0,0,0,-1,0,1,-1,-1,1,1,-1,-1,-1],
    'W': [-1,1,-1,1,1,-1,0,0,1,1,-1,1,-1,-1,0,1,-1,-1,1,-1,-1,-1,-1],
    'Y': [-1,1,-1,1,-1,-1,0,0,1,1,-1,-1,-1,-1,0,1,-1,-1,1,-1,-1,-1,-1],
    'Z': [1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,1,1,-1,-1,-1],
    'ZH': [1,-1,-1,-1,0,1,-1,1,-1,0,0,0,0,-1,0,1,-1,-1,1,-1,-1,-1,-1],
}

#############################
### DEFINE HELPER CLASSES ###
#############################

class Word(object):
    '''
    A word is a grapheme-phoneme pair
      grapheme - str
      phoneme  - list(str)
      phonetic_features - numpy_array(int) [n_subphones x n_features]
      phonemes are assumed to be stress-less
    '''
    def __init__(self, grapheme, phoneme, clean_grapheme=False, clean_phoneme=True):
        self.grapheme = Word.get_clean_grapheme(grapheme) if clean_grapheme else grapheme
        self.phoneme = Word.get_clean_phoneme(phoneme) if clean_phoneme else phoneme
        self.phonological_phoneme = Word.get_phonological_phoneme(phoneme)
        self.phonological_features = Word.get_phonological_features(phoneme)

    def get_neighbor_words(self, include_self=True):
        '''
        Find neighbors using naive Word2Vec nearest-neighbors

        Do NOT mess around with casing (can change this later)
        Do NOT mess aroung with lemmatization (can change this later)
        Do NOT mess aroung with multiple pronounciations (can change this later)

        TODO: update this to use WordNet distance, or something else more intelligent

        TODO: Find 50 nearest neighbors, then 50 nearest neighbors of neighbors
        This will yield more closely semantially-related words because Word2Vec is only meaningful over very short distances

        '''
        neighbor_graphemes = self.get_semantic_neighbor_graphemes(is_recursive=False)

        neighbor_words = []
        for grapheme in neighbor_graphemes:
            if grapheme in grapheme_to_phoneme_dict:
                phoneme = grapheme_to_phoneme_dict[grapheme][0]
                neighbor_words.append(Word(grapheme, phoneme))

        if include_self:
            neighbor_words.append(self)

        return neighbor_words

    def get_semantic_neighbor_graphemes(self, is_recursive=False, max_neighbors=100, max_recursive_neighbors=50):
        '''
        Find semantic neighbor graphemes using word2vec

        is_recursive = False : find the max_neighbors nearest neighbors to self.grapheme
        is_recursive = True  : find the max_recursive_neighbors nearest neighbors to self.grapheme, and then find the max_recursive_neighbors to *those*, removing duplicates

        TODO: Remove recursion logic
        '''
        if is_recursive:
            return Word.get_word2vec_neighbors(self.grapheme, max_neighbors)
        else:
            neighbor_graphemes = Word.get_word2vec_neighbors(self.grapheme, max_recursive_neighbors)
            for neighbor_grapheme in neighbor_graphemes[:]:
                neighbor_graphemes += Word.get_word2vec_neighbors(neighbor_grapheme, max_recursive_neighbors)
            # Remove duplicates, and remove original grapheme if present
            neighbor_graphemes = list(set(neighbor_graphemes))
            if self.grapheme in neighbor_graphemes:
                neighbor_graphemes.remove(self.grapheme)
            return neighbor_graphemes

    @staticmethod
    def get_word2vec_neighbors(grapheme, num_neighbors):
        '''
        Get the num_neighbors nearest neighbors to 'grapheme', as given by the global word2vec_model
        '''
        return list(zip(*word2vec_model.most_similar(positive=[grapheme], topn=num_neighbors))[0])

    @staticmethod
    def get_clean_grapheme(grapheme):
        '''
        Convert grapheme to string, and make lower-case
        '''
        return str(grapheme).lower()

    @staticmethod
    def get_clean_phoneme(phoneme):
        '''
        Remove non-alpha characters denoting stress and other non-phonetic information
        '''
        return [filter(str.isalpha, str(phone)) for phone in phoneme]

    @staticmethod
    def get_phonological_phoneme(phoneme):
        '''
        Convert ARPABET-type phoneme to phonological-type phoneme

        A phonological-type phoneme is a phoneme which breaks apart diphthongs and rhotics into sub-phones
        This representation can then be directly mapped onto phonological feature vectors
        '''
        arpabet_phoneme = Word.get_clean_phoneme(phoneme)
        phonological_phone_list = [ARPABET_PHONE_TO_PHONOLOGICAL_PHONE_DICT[phone] for phone in arpabet_phoneme]
        return reduce(lambda x,y: x+y, phonological_phone_list)

    @staticmethod
    def get_phonological_features(phoneme):
        '''
        Convert phoneme to phonological [n_phones x n_features] feature array
        '''
        phonological_phoneme = Word.get_phonological_phoneme(phoneme)
        feature_list = [PHONOLOGICAL_PHONE_TO_PHONOLOGICAL_FEATURE_DICT[phone] for phone in phonological_phoneme]
        return np.array(feature_list)


class Pun(object):
    '''
    A pun carries with it information about its quality to allow for later filtering

    MIN_LENGTH : minimum number of phones each phonemes must have
    MIN_OVERLAP : minimum number of phones which must overlap between the phonemes
    MAX_NORMED_DISTANCE : min(|overlapping_phones| / |total_phones|)
    MIN_NORMED_OVERLAP : min(|phonetic_distance| / |total_phones|)
    '''

    MIN_LENGTH = 3
    MIN_OVERLAP = 3
    MAX_NORMED_DISTANCE = 0.4
    MIN_NORMED_OVERLAP = 0.4

    def __init__(self, word1, word2, phonological_overlap1=None, phonological_overlap2=None, phoneme_distance=None, normed_phoneme_distance=None, overlap_size=None, normed_overlap_size=None):
        self.word1 = word1
        self.word2 = word2
        self.phonological_overlap1 = phonological_overlap1
        self.phonological_overlap2 = phonological_overlap2
        self.phoneme_distance = phoneme_distance
        self.normed_phoneme_distance = normed_phoneme_distance
        self.overlap_size = overlap_size
        self.normed_overlap_size = normed_overlap_size

    def generate_pun(self):
        '''
        Find best-quality phonetic overlap (according to class quality ordering) which satisfles the allotted constraints
        '''

        # Only attempt to generate a pun if each word is long enough
        if len(self.word1.phoneme) < Pun.MIN_LENGTH or len(self.word2.phoneme) < Pun.MIN_LENGTH:
            self.is_pun = False
            return

        # For each allowable window size, consider all combinations of subphones
        min_phonological_features = min([len(self.word1.phonological_features), len(self.word2.phonological_features)])
        max_phonological_features = max([len(self.word1.phonological_features), len(self.word2.phonological_features)])
        for overlap in range(Pun.MIN_OVERLAP, min_phonological_features + 1):
            for i1 in range(len(self.word1.phonological_features) - overlap + 1):
                for i2 in range(len(self.word2.phonological_features) - overlap + 1):
                    phonological_overlap1 = self.word1.phonological_phoneme[i1:i1+overlap]
                    phonological_overlap2 = self.word2.phonological_phoneme[i2:i2+overlap]
                    phoneme_distance = Pun.phonological_distance(self.word1.phonological_features[i1:i1+overlap], self.word2.phonological_features[i2:i2+overlap])
                    normed_phoneme_distance = 1.0 * phoneme_distance / max_phonological_features
                    normed_overlap_size = 1.0 * overlap / max_phonological_features
                    this_pun = Pun(self.word1, self.word2, phonological_overlap1=phonological_overlap1, phonological_overlap2=phonological_overlap2, phoneme_distance=phoneme_distance, normed_phoneme_distance=normed_phoneme_distance, overlap_size=overlap, normed_overlap_size=normed_overlap_size)      
                    # TODO: initialize with first quality specs
                    if self.phoneme_distance is None or this_pun > self: # no distances yet computed
                        self.phonological_overlap1 = phonological_overlap1
                        self.phonological_overlap2 = phonological_overlap2
                        self.phoneme_distance = phoneme_distance
                        self.normed_phoneme_distance = normed_phoneme_distance
                        self.overlap_size = overlap
                        self.normed_overlap_size = normed_overlap_size

    def is_valid_pun(self):
        if min(len(self.word1.phoneme), len(self.word1.phoneme)) >= Pun.MIN_LENGTH and self.overlap_size >= Pun.MIN_OVERLAP and self.normed_phoneme_distance <= Pun.MAX_NORMED_DISTANCE and self.normed_overlap_size >= Pun.MIN_NORMED_OVERLAP:
            return True
        else:
            return False

    def pun_quality_tuple(self):
        # high, low, high, low
        # Ok... ording logic is fucked; really, just need to hold onto whichever one(s) allow me to satisfy the 'is_valid_pun' constraints
        return (-self.normed_phoneme_distance, self.overlap_size, -self.phoneme_distance, self.normed_overlap_size)

    def __gt__(self, another_pun):
        '''
        Better off performing an alignment with respect to a loss here...

        First check if one of the puns meets the minimum sucess criteria; if so, then rank it higher

        If both (or neither) meet the success criteria, then rank them according to tuple order
        '''
        if self.is_valid_pun() and not another_pun.is_valid_pun():
            return True
        elif not self.is_valid_pun() and another_pun.is_valid_pun():
            return False
        else:
            return self.pun_quality_tuple() > another_pun.pun_quality_tuple()
 
    def __lt__(self, another_pun):
        '''
        Better off performing an alignment with respect to a loss here...

        First check if one of the puns meets the minimum sucess criteria; if so, then rank it higher

        If both (or neither) meet the success criteria, then rank them according to tuple order
        '''
        if self.is_valid_pun() and not another_pun.is_valid_pun():
            return False
        elif not self.is_valid_pun() and another_pun.is_valid_pun():
            return True
        else:
            return self.pun_quality_tuple() < another_pun.pun_quality_tuple()

    def __eq__(self, another_pun):
        return self.pun_quality_tuple() == another_pun.pun_quality_tuple()

    def __str__(self):
        # TODO: order words based on gramatical rules (adj noun, adv adj, etc)
        return '\n{} {}\n({} / {})\nNormed Distance:\t{:.02}\nNormed Overlap:\t\t{:.02}\nAbsolute Distance:\t{}\nAbsolute Overlap:\t{}'.format(self.word1.grapheme, self.word2.grapheme, '-'.join(self.phonological_overlap1), '-'.join(self.phonological_overlap2), self.normed_phoneme_distance, self.normed_overlap_size, self.phoneme_distance, self.overlap_size)

    @staticmethod
    def phonological_distance(feature_array1, feature_array2):
        '''
        Phoneme distance is the sum of the constituent phone distances
        '''
        if len(feature_array1) == len(feature_array2):
            return abs(feature_array1 - feature_array2).sum()
        else:
            raise Exception('Feature arrays must be the same length: {} != {}'.format(len(feature_array1), len(feature_array2)))


###############################
### DEFINE HELPER FUNCTIONS ###
###############################

def validate_input(input_string):
    '''
    Verify that the input is comprised of two words, and that both words are present in Word2Vec
    '''
    if type(input_string) is not str or len(input_string.split()) != 2:
        status = 1
        message = 'Error: Input should be of the form "[word1] [word2]"'
    else:
        grapheme1, grapheme2 = input_string.split()

        grapheme1_exists = grapheme1 in word2vec_model.vocab
        phoneme1_exists = grapheme1 in grapheme_to_phoneme_dict
        grapheme2_exists = grapheme2 in word2vec_model.vocab
        phoneme2_exists = grapheme2 in grapheme_to_phoneme_dict

        if not (grapheme1_exists and phoneme1_exists) and (grapheme2_exists and phoneme2_exists):
            status = 1
            message = "Error: '{}' is not a recognized word, please check the spelling and capitalization".format(grapheme1)
        elif (grapheme1_exists and phoneme1_exists) and not (grapheme2_exists and phoneme2_exists):
            status = 1
            message = "Error: '{}' is not a recognized word, please check the spelling and capitalization".format(grapheme2)
        elif not (grapheme1_exists and phoneme1_exists) and not (grapheme2_exists and phoneme2_exists):
            status = 1
            message = "Error: neithor '{}' nor '{}' is not a recognized word, please check the spelling and capitalization".format(grapheme1, grapheme2)
        else:
            status = 0
            message = ''

    return status, message


def parse_input(input_string):
    grapheme1, grapheme2 = input_string.split()
    phoneme1 = grapheme_to_phoneme_dict[grapheme1][0]
    phoneme2 = grapheme_to_phoneme_dict[grapheme2][0]
    return Word(grapheme1, phoneme1), Word(grapheme2, phoneme2)


def get_valid_puns(words1_neighbors, words2_neighbors, ordered=True, is_test=False):
    pun_list = []
    for neighbor1 in words1_neighbors:
        for neighbor2 in words2_neighbors:
            this_pun = Pun(neighbor1, neighbor2)
            this_pun.generate_pun()
            if this_pun.is_valid_pun() or is_test:
                pun_list.append(this_pun)

    # sort using the natural quality ordering
    if ordered:
        pun_list.sort(reverse=True)

    return pun_list


if __name__ == '__main__':

    # Parse input args
    if '--testing' in sys.argv:
        is_test = True
    else:
        is_test = False

    if not is_test:
        # Load Google's pre-trained Word2Vec model
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)  

    # Load CMUdict phonetic dictionary
    grapheme_to_phoneme_dict = nltk.corpus.cmudict.dict()

    while True:
        input_string = raw_input("\nSeed words:  ")

        if not is_test:
            status, message = validate_input(input_string)
            # If anything is wrong with the input, set the status to 1, print a message describing the problem, and skip the rest of the logic
            if status == 1:
                print message
                continue

        word1, word2 = parse_input(input_string)

        if not is_test:
            nearest_words1 = word1.get_neighbor_words()
            nearest_words2 = word2.get_neighbor_words()
        else:
            nearest_words1 = [word1]
            nearest_words2 = [word2]

        pun_list = get_valid_puns(nearest_words1, nearest_words2, is_test=is_test)

        for i, pun in enumerate(pun_list):
            if i >= MAX_PUNS:
                break
            print pun
