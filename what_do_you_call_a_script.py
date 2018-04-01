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

ARPABET_PHONE_TO_PHONOLOGICAL_PHONE_DICT = {
    'AA': ['AA']
    'AE': ['AE']
    'AH': ['AH']
    'AO': ['AO']
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

    def get_semantic_neighbor_graphemes(self, is_recursive=False, max_neighbors=300, max_recursive_neighbors=50):
        '''
        Find semantic neighbor graphemes using word2vec

        is_recursive = False : find the max_neighbors nearest neighbors to self.grapheme
        is_recursive = True  : find the max_recursive_neighbors nearest neighbors to self.grapheme, and then find the max_recursive_neighbors to *those*, removing duplicates
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
        arpabet_phoneme = get_clean_phoneme(phoneme)
        phonological_phone_list = [ARPABET_PHONE_TO_PHONOLOGICAL_PHONE_DICT[phone] for phone in arpabet_phoneme]
        return reduce(lambda x,y: x+y, phonological_phone_list)

    @staticmethod
    def get_phonological_features(phoneme):
        '''
        Convert phoneme to phonological [n_phones x n_features] feature array
        '''
        phonological_phoneme = get_phonological_phoneme(phoneme)
        feature_list = [PHONOLOGICAL_PHONE_TO_PHONOLOGICAL_FEATURE_DICT[phone] for phone in phonological_phoneme]
        return np.array(feature_list)


class Pun(object):
    '''
    A pun is a combination of two words which is phonetically acceptable

    A pun carries with it information about its quality to allow for later filtering
    '''

    # def __init__(self, word1, word2, is_pun=None, is_portmanteau=None, is_rhyme=None, is_pun=None, pun_string=None, 
    def __init__(self, word1, word2, is_portmanteau=None, is_rhyme=None, is_pun=None, pun_string=None, 
                 phoneme_distance=None, normed_phoneme_distance=None, overlap_size=None, normed_overlap_size=None):
        self.word1 = word1
        self.word2 = word2
        
        # self.is_pun = is_pun
        self.is_portmanteau = is_portmanteau
        self.is_rhyme = is_rhyme

        self.pun_string = pun_string

        self.phoneme_distance = phoneme_distance
        self.normed_phoneme_distance = normed_phoneme_distance
        self.overlap_size = overlap_size
        self.normed_overlap_size = normed_overlap_size

    def generate_pun_string(self, min_length=3, min_overlap=3, max_normed_distance=0.4, min_normed_overlap=0.4):
        '''
        Find max-length phonetic overlap which satisfles the allotted constraints

        min_length : minimum number of phones each phonemes must have
        min_overlap : minimum number of phones which must overlap between the phonemes
        min_normed_distance : min(|overlapping_phones| / |total_phones|)
        min_normed_overlap : min(|phonetic_distance| / |total_phones|)
        '''

        # Only attempt to generate a pun if each word is long enough
        if len(self.word1.phoneme) < min_length or len(self.word2.phoneme) < min_length:
            # self.is_pun = False
            self.is_portmanteau = False
            self.is_rhyme = False
            return

        # For each allowable window size, consider all combinations of subphones
        min_phonological_features = min([len(self.word1.phonological_features), len(self.word2.phonological_features)])
        max_phonological_features = min([len(self.word1.phonological_features), len(self.word2.phonological_features)])
        pun_list = []
        for overlap in range(min_overlap, min_phonological_features + 1):
            for i1 in range(len(self.word1.phonological_features) - overlap + 1):
                for i2 in range(len(self.word2.phonological_features) - overlap + 1):
                    phoneme_distance = Pun.phonological_distance(self.word1.phonological_features[i:j], self.word2.phonological_features[k:l])
                    normed_phoneme_distance = 1.0 * phoneme_distance / max_phonological_features
                    normed_overlap_size = 1.0 * overlap / max_phonological_features
                    # TODO : verify off-by-one logic
                    if len(self.word1.phonological_features) - overlap == i1 and i2 == 0:
                        is_portmanteau = True
                    else 
                        is_portmanteau = False

                    if len(self.word1.phonological_features) - overlap == i1 and len(self.word2.phonological_features) - overlap == i2:
                        is_rhyme = True
                    else 
                        is_rhyme = False

                        Pun(None, None, )


        # Keep as much of each word preserved as possible, i.e. return as soon as an acceptable overlap is found
        for overlap in range(min_overlap, max_overlap + 1):
            if self.word1.phoneme[len(self.word1.phoneme) - overlap:] == self.word2.phoneme[:overlap]:
                # At least one of the overlapping phones must be a vowel
                overlap_phoneme = Word.get_clean_phoneme(self.word2.phoneme[:overlap])
                if len(set(overlap_phoneme) & ARPABET_VOWELS) >= 1:
                    self.has_portmanteau = True
                    portmanteau_phoneme = self.word1.phoneme + self.word2.phoneme[overlap:]
                    portmanteau_grapheme = self.get_portmanteau_grapheme(overlap)
                    self.portmanteau_word = Word(portmanteau_grapheme, portmanteau_phoneme)
                    return

        self.has_portmanteau = False
        

    @staticmethod
    def phonological_distance(feature_array1, feature_array2):
        '''
        Phoneme distance is the sum of the constituent phone distances
        '''
        if len(phoneme1) == len(phoneme2):
            return abs(feature_array1 - feature_array2).sum()
        else:
            raise Exception('Feature arrays must be the same length: {} != {}'.format(len(phoneme1), len(phoneme2)))


class Portmanteau(object):
    '''
    A portmanteau is a combination of two words which is both graphemically and phonetically acceptable

    For a given pair of words, zero, one, or many acceptible portmanteaus can be possible

    If multiple acceptable portmanteaus are found, pick the first one that's found
    By default this will favor shorter overlaps over longer ones, and favor the order word1-word2 over word2-word1
    '''

    # min_length = 3      # minimum number of phones each phonemes must have
    # min_overlap = 3     # minimum number of phones which must overlap between the phonemes
    # min_nonoverlap = 1  # minimum number of phones which must NOT overlap for each of the phonemes

    def __init__(self, word1, word2):
        self.word1 = word1
        self.word2 = word2
        self.has_portmanteau = None
        self.portmanteau_word = None

    def generate_portmanteau_word(self, min_length=3, min_overlap=3, min_nonoverlap=1):
        '''
        Generate a portmanteau of the member words (if possible)
        '''

        # Only attempt to generate a portmanteau if each phoneme is long enough
        if len(self.word1.phoneme) < min_length or len(self.word2.phoneme) < min_length:
            self.has_portmanteau = False
            return

        max_overlap = min(len(self.word1.phoneme), len(self.word2.phoneme)) - min_nonoverlap

        for word_order in ('forward','backward'):
            # If the word order is 'backward', then flip the variable names
            if word_order == 'backward':
                self.word1, self.word2 = self.word2, self.word1

            # Keep as much of each word preserved as possible, i.e. return as soon as an acceptable overlap is found
            for overlap in range(min_overlap, max_overlap + 1):
                if self.word1.phoneme[len(self.word1.phoneme) - overlap:] == self.word2.phoneme[:overlap]:
                    # At least one of the overlapping phones must be a vowel
                    overlap_phoneme = Word.get_clean_phoneme(self.word2.phoneme[:overlap])
                    if len(set(overlap_phoneme) & ARPABET_VOWELS) >= 1:
                        self.has_portmanteau = True
                        portmanteau_phoneme = self.word1.phoneme + self.word2.phoneme[overlap:]
                        portmanteau_grapheme = self.get_portmanteau_grapheme(overlap)
                        self.portmanteau_word = Word(portmanteau_grapheme, portmanteau_phoneme)
                        return

        self.has_portmanteau = False

    def get_portmanteau_grapheme(self, overlap):
        '''
        TODO: Implement this function with the help of a premade phoneme/grapheme alignment algorithm

        # Use one of these two packages for phoneme/grapheme alignment:
        # 1) https://github.com/AdolfVonKleist/Phonetisaurus
        # 2) https://github.com/letter-to-phoneme/m2m-aligner

        For now use the trival hyphenated grapheme combination
        '''
        return self.word1.grapheme + '-' + self.word2.grapheme

    def print_portmanteau(self):
        '''
        Print the resulting portmaneau
        '''
        if self.has_portmanteau:
            print '{}  ({} + {})'.format(self.portmanteau_word.grapheme, self.word1.grapheme, self.word2.grapheme)
        else:
            print "Could not find portmanteau of '{}' and '{}'".format(self.word1.grapheme, self.word2.grapheme)


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


def get_valid_portmanteaus(words1_neighbors, words2_neighbors):
    portmanteau_list = []
    for neighbor1 in words1_neighbors:
        for neighbor2 in words2_neighbors:
            this_portmanteau = Portmanteau(neighbor1, neighbor2)
            this_portmanteau.generate_portmanteau_word()
            # TODO: change 'has_portmanteau' from a method to a function
            if this_portmanteau.has_portmanteau:
                portmanteau_list.append(this_portmanteau)

    return portmanteau_list


if __name__ == '__main__':

    # Load Google's pre-trained Word2Vec model
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)  

    # Load CMUdict phonetic dictionary
    grapheme_to_phoneme_dict = nltk.corpus.cmudict.dict()

    while True:
        input_string = raw_input("\nSeed words:  ")
        # input_string = "labrador dormitory"

        status, message = validate_input(input_string)

        # If anything is wrong with the input, set the status to 1, print a message describing the problem, and skip the rest of the logic
        if status == 1:
            print message
            continue

        word1, word2 = parse_input(input_string)

        nearest_words1 = word1.get_neighbor_words()
        nearest_words2 = word2.get_neighbor_words()

        portmanteau_list = get_valid_portmanteaus(nearest_words1, nearest_words2)

        for portmanteau in portmanteau_list:
            portmanteau.print_portmanteau()
