import gensim
import nltk
import numpy as np


#############################
### DEFINE HELPER CLASSES ###
#############################

class Word:
    '''
    A word is a grapheme-phoneme pair
      grapheme - str
      phoneme  - list(str)

      phonemes are assumed to be stress-less
    '''
    def __init__(self, grapheme, phoneme, clean_grapheme=False, clean_phoneme=True):
        self.grapheme = self.get_clean_grapheme(grapheme) if clean_grapheme else grapheme
        self.phoneme = self.get_clean_phoneme(phoneme) if clean_phoneme else phoneme

    def get_neighbor_words(self, include_self=True, max_neighbors=300):
        '''
        Find neighbors using naive Word2Vec nearest-neighbors

        Do NOT mess around with casing (can change this later)
        Do NOT mess aroung with lemmatization (can change this later)
        Do NOT mess aroung with multiple pronounciations (can change this later)

        TODO: update this to use WordNet distance, or something else more intelligent
        '''
        neighbor_graphemes = list(zip(*word2vec_model.most_similar(positive=[self.grapheme], topn=max_neighbors))[0])

        neighbor_words = []
        for grapheme in neighbor_graphemes:
            if grapheme in grapheme_to_phoneme_dict:
                phoneme = grapheme_to_phoneme_dict[grapheme][0]
                neighbor_words.append(Word(grapheme, phoneme))

        if include_self:
            neighbor_words.append(self)

        return neighbor_words

    def get_clean_grapheme(self, grapheme):
        '''
        Convert grapheme to string, and make lower-case
        '''
        return str(grapheme).lower()


    def get_clean_phoneme(self, phoneme):
        '''
        Remove non-alpha characters denoting stress and other non-phonetic information
        '''
        return [filter(str.isalpha, str(phone)) for phone in phoneme]


class Portmanteau:
    '''
    A portmanteau is a combination of two words which is both graphemically and phonetically acceptable

    For a given pair of words, zero, one, or many acceptible portmanteaus can be possible

    If multiple acceptable portmanteaus are found, pick the first one that's found
    By default this will favor shorter overlaps over longer ones, and favor the order word1-word2 over word2-word1
    '''

    # min_length = 3      # minimum number of phones each phonemes must have
    # min_overlap = 2     # minimum number of phones which must overlap between the phonemes
    # min_nonoverlap = 1  # minimum number of phones which must NOT overlap for each of the phonemes

    def __init__(self, word1, word2):
        self.word1 = word1
        self.word2 = word2
        self.has_portmanteau = None
        self.portmanteau_word = None

    def generate_portmanteau_word(self, min_length=3, min_overlap=2, min_nonoverlap=1):
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

    # # Load Google's pre-trained Word2Vec model
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
