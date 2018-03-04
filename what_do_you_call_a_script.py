import gensim
import nltk
import numpy as np


###############################
### DEFINE GLOBAL CONSTANTS ###
###############################

# Load Google's pre-trained Word2Vec model
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/jonathansimon/code/Hackathon_3-1-18/GoogleNews-vectors-negative300.bin.gz', binary=True)  

# Load CMUdict phonetic dictionary
corpus = nltk.corpus.cmudict.dict()

# Load the (subpar) nltk lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# ARPAbet to phonetic features mapping, using phonetic feature chart here: http://www.artoflanguageinvention.com/papers/features.pdf
arpabet_to_feature_vec = {
    'AA': np.array([-1,1,1,1,-1,-1,0,0,1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # variant
    'AE': np.array([-1,1,1,1,-1,-1,0,0,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1]),
    # 'AH': np.array([-1,1,1,1,1,-1,0,0,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # rounding
    'AH': np.array([-1,1,1,1,1,-1,0,0,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # treat as UH rather than AO
    'AO': np.array([-1,1,1,1,1,-1,0,0,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]),

    'AW': [], # diphthong
    'AW1': np.array([-1,1,1,1,-1,-1,0,0,1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # = AA
    'AW2': np.array([-1,1,1,1,1,-1,0,0,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # = UH
        
    'AY': [], # diphthong
    'AY1': np.array([-1,1,1,1,-1,-1,0,0,1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # = AA
    'AY2': np.array([-1,1,1,1,-1,-1,0,0,1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # = IH
        
    'B': np.array([1,-1,-1,1,-1,-1,0,0,-1,0,0,0,0,-1,0,1,-1,-1,-1,-1,-1,-1,-1]),
    'CH': np.array([1,-1,-1,-1,0,1,-1,1,1,1,-1,-1,-1,-1,0,-1,-1,-1,0,1,-1,1,-1]),
    'D': np.array([1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,-1,-1,-1,-1,-1]),
    'DH': np.array([1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,1,-1,-1,-1,-1]), # FIXED
    'EH': np.array([-1,1,1,1,-1,-1,0,0,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]),
        
    'ER': [], # over-compensating for rhoticity
    'ER1': np.array([-1,1,1,1,1,-1,0,0,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # = AO
    'ER2': np.array([1,1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,1,-1,-1,-1,-1]), # = R
        
    'EY': [], # diphthong
    'EY1': np.array([-1,1,1,1,-1,-1,0,0,1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,-1]),
    'EY2': np.array([-1,1,1,1,-1,-1,0,0,1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # = IH
        
    'F': np.array([1,-1,-1,1,-1,-1,0,0,-1,0,0,0,0,-1,0,-1,-1,-1,1,1,-1,-1,-1]),
    'G': np.array([1,-1,-1,-1,0,-1,0,0,1,1,-1,1,-1,-1,0,1,-1,-1,-1,-1,-1,-1,-1]),
    'HH': np.array([-1,-1,-1,-1,0,-1,0,0,-1,0,0,0,0,-1,0,-1,1,-1,1,-1,-1,-1,-1]),
    'IH': np.array([-1,1,1,1,-1,-1,0,0,1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]),
    'IY': np.array([-1,1,1,1,-1,-1,0,0,1,1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,-1]),
    'JH': np.array([1,-1,-1,-1,0,1,-1,1,1,1,-1,-1,-1,-1,0,1,-1,-1,0,1,-1,1,-1]),
    'K': np.array([1,-1,-1,-1,0,-1,0,0,1,1,-1,1,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1]),
    'L': np.array([1,1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,1,-1,1,-1,-1]),
    'M': np.array([1,1,-1,1,-1,-1,0,0,-1,0,0,0,0,-1,0,1,-1,-1,-1,-1,-1,-1,1]),
    'N': np.array([1,1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,-1,-1,-1,-1,1]),
    'NG': np.array([1,1,-1,-1,0,-1,0,0,1,1,-1,1,-1,-1,0,1,-1,-1,-1,-1,-1,-1,1]),
        
    'OW': [], # diphthong
    'OW1': np.array([-1,1,1,1,1,-1,0,0,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1]),
    'OW2': np.array([-1,1,1,1,1,-1,0,0,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # = UH
        
    'OY': [], # diphthong
    'OY1': np.array([-1,1,1,1,1,-1,0,0,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # = AO
    'OY2': np.array([-1,1,1,1,-1,-1,0,0,1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]), # = IH
        
    'P': np.array([1,-1,-1,1,-1,-1,0,0,-1,0,0,0,0,-1,0,-1,-1,-1,-1,-1,-1,-1,-1]),
    'R': np.array([1,1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,1,-1,-1,-1,-1]),
    'S': np.array([1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,-1,-1,-1,1,1,-1,-1,-1]),
    'SH': np.array([1,-1,-1,-1,0,1,-1,1,-1,0,0,0,0,-1,0,-1,-1,-1,1,-1,-1,-1,-1]),
    'T': np.array([1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,-1,-1,-1,-1,-1,-1,-1,-1]),
    'TH': np.array([1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,-1,-1,-1,1,-1,-1,-1,-1]),
    'UH': np.array([-1,1,1,1,1,-1,0,0,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]),
    'UW': np.array([-1,1,1,1,1,-1,0,0,1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1]),
    'V': np.array([1,-1,-1,1,-1,-1,0,0,-1,0,0,0,0,-1,0,1,-1,-1,1,1,-1,-1,-1]),
    'W': np.array([-1,1,-1,1,1,-1,0,0,1,1,-1,1,-1,-1,0,1,-1,-1,1,-1,-1,-1,-1]),
    'Y': np.array([-1,1,-1,1,-1,-1,0,0,1,1,-1,-1,-1,-1,0,1,-1,-1,1,-1,-1,-1,-1]),
    'Z': np.array([1,-1,-1,-1,0,1,1,-1,-1,0,0,0,0,-1,0,1,-1,-1,1,1,-1,-1,-1]),
    'ZH': np.array([1,-1,-1,-1,0,1,-1,1,-1,0,0,0,0,-1,0,1,-1,-1,1,-1,-1,-1,-1]),
}

# ER is not a diphthong, but just leave it in here for now for simplicity
diphthongs = set(['AW', 'AY', 'EY', 'OW', 'OY', 'ER'])


###############################
### DEFINE HELPER FUNCTIONS ###
###############################

def clean_words(initial_word, word_list):
    '''
    Given a seed word and a list of related words, aggregate them and clean the results
    '''
    # Add the seed word to the word list
    clean_word_list = [initial_word] + list(word_list)
    # Make the words lowercase
    clean_word_list = [x.lower() for x in clean_word_list]
    # Lemmatize the words
    clean_word_list = [lemmatizer.lemmatize(x) for x in clean_word_list]
    # Remove duplicates
    clean_word_list = list(set(clean_word_list))
    # Remove all words which are not present in the cmudict
    clean_word_list = filter(lambda x: x in corpus, clean_word_list)
    # Return the result as a list
    return clean_word_list


def clean_phoneme(phone_list):
    '''
    Given a phoneme (represented as a list of phones), verify that all phones are present in the ARPAbet mapper,
    and map diphthongs (and rhotics) to pairs of sub-phones as needed
    '''
    new_phone_list = []
    for phone in phone_list:
        clean_phone = filter(str.isalpha, str(phone))
        if clean_phone in arpabet_to_feature_vec:
            if clean_phone in diphthongs:
                new_phone_list.append(clean_phone+'1')
                new_phone_list.append(clean_phone+'2')
            else:
                new_phone_list.append(clean_phone)
        else:
            raise Exception('Phone is not in ARPABET: {}'.format(clean_phone))
    return new_phone_list


def phone_distance(phone1, phone2):
    '''
    Measure distance between phones as the manhattan distance between their feature vectors
    '''
    return np.abs(arpabet_to_feature_vec[phone1] - arpabet_to_feature_vec[phone2]).sum()


def phoneme_distance(phoneme1, phoneme2):
    '''
    Phoneme distance is the sum of the constituent phone distances
    '''
    if len(phoneme1) == len(phoneme2):
        distance = 0
        for i in range(len(phoneme1)):
            distance += phone_distance(phoneme1[i], phoneme2[i])
        return distance
    else:
        raise Exception('Phonemes must be the same length: {}, {}'.format(phoneme1, phoneme2))


# Rules:
# - max phones which can be discarded per word: 1
# - min total phones per word: 4
# - only *one* word can be truncated
# - distance between the overlapping phonemes must be <= |overlap|/2.0
# - min overlapping phoneme length in phones: 3
def get_portmanteaus(raw_phoneme1, raw_phoneme2, word1, word2, min_phoneme_length=4, max_discarded=1, min_overlap=3, min_normed_distance=0.5):
    '''
    Given two phonemes, generate phonetically plausible mergers of the two
    '''

    portmanteau_list = []

    if len(raw_phoneme1) < min_phoneme_length or len(raw_phoneme2) < min_phoneme_length:
        return portmanteau_list
    
    phoneme1 = clean_phoneme(raw_phoneme1)
    phoneme2 = clean_phoneme(raw_phoneme2)
    
    # Case 1: end of first phoneme is truncated
    for n_phones_truncated in range(max_discarded+1): 
        phoneme1_trunc = phoneme1[:len(phoneme1)-n_phones_truncated]
        for overlap_size in range(min_overlap,1+min(len(phoneme1_trunc),len(phoneme2))):
            phoneme_chunk1 = phoneme1_trunc[len(phoneme1_trunc)-overlap_size:]
            phoneme_chunk2 = phoneme2[:overlap_size]
            
            distance = phoneme_distance(phoneme_chunk1, phoneme_chunk2)
            normed_distance = 1.0*distance / len(phoneme_chunk1)
            # the overlap is phonetically good enough (accounting for length) to add to our list
            if normed_distance <= min_normed_distance:
                this_portmanteau_phoneme = '-'.join(phoneme1_trunc + phoneme2[overlap_size:])
                this_portmanteau = {'portmanteau_phoneme': this_portmanteau_phoneme, 'first_word': word1, 'second_word': word2, 'overlap_size': overlap_size, 'phonetic_distance': distance, 'normed_phonetic_distance': normed_distance}
                portmanteau_list.append(this_portmanteau)
        
    
    # Case 2: start of second phoneme is truncated
    for n_phones_truncated in range(max_discarded+1):
        phoneme2_trunc = phoneme2[n_phones_truncated:]
        for overlap_size in range(min_overlap,1+min(len(phoneme1),len(phoneme2_trunc))):
            phoneme_chunk1 = phoneme1[len(phoneme1)-overlap_size:]
            phoneme_chunk2 = phoneme2_trunc[:overlap_size]
            
            distance = phoneme_distance(phoneme_chunk1, phoneme_chunk2)
            normed_distance = 1.0*distance / len(phoneme_chunk1)
            # the overlap is phonetically good enough (accounting for length) to add to our list
            if normed_distance <= min_normed_distance:
                this_portmanteau_phoneme = '-'.join(phoneme1 + phoneme2_trunc[overlap_size:])
                this_portmanteau = {'portmanteau_phoneme': this_portmanteau_phoneme, 'first_word': word1, 'second_word': word2, 'overlap_size': overlap_size, 'phonetic_distance': distance, 'normed_phonetic_distance': normed_distance}
                portmanteau_list.append(this_portmanteau)
    
    return portmanteau_list


def get_all_portmanteaus(nearest_words1, nearest_words2):
    portmanteau_list = []
    for w1 in nearest_words1:
        for w2 in nearest_words2:
            if w1 == w2:
                continue
            phoneme_list1 = corpus[w1]
            phoneme_list2 = corpus[w2]
            for p1 in phoneme_list1:
                for p2 in phoneme_list2:
                    # Input order matters, so call the function twice with inputs flipped
                    portmanteaus = get_portmanteaus(p1, p2, w1, w2) + get_portmanteaus(p2, p1, w2, w1)
                    portmanteau_list.extend(portmanteaus)

    return portmanteau_list


def clean_and_sort_results(portmanteau_list):
    '''
    Remove phonetic duplicates from the results list, keeping only the version with the largest overlap
    '''

    # find unique phonemes
    unique_phonemes = set(map(lambda x: x['portmanteau_phoneme'], portmanteau_list))

    # get unique results, picking largest overlap with the the smallest normed distance as the winner
    cleaned_results = []
    for phoneme in unique_phonemes:
        these_dicts = []
        for this_portmanteau_dict in portmanteau_list:
            if this_portmanteau_dict['portmanteau_phoneme'] == phoneme:
                these_dicts.append(this_portmanteau_dict)
        if these_dicts:
            winner_tuple = sorted(these_dicts, key=lambda x: (-x['overlap_size'],x['normed_phonetic_distance']))[0]
            cleaned_results.append(winner_tuple)

    # remove duplicates (very roundabout, but necessary because dicts are unhashable)
    # portmanteau_list = [dict(tupleized) for tupleized in set(tuple(item.items()) for item in portmanteau_list)]

    # sort the remains
    cleaned_results.sort(key=lambda x: (x['normed_phonetic_distance'],-x['overlap_size']))
    # portmanteau_list.sort(key = lambda x: x['overlap_size'], reverse=True)

    return cleaned_results


if __name__ == '__main__':
  while True:
    # List of pairs that produced reasonable results:
    # 1) booking speed --> travelocity
    # 2) hostel dogs --> labradormitory
    # 3) amazing butler --> chauffeurwhelming
    # 4) quark barbell --> electricepts
    # 5) Beethoven hotel --> Raddisonata
    # 6) violent apartment --> vigilantelord
    # 7) sad airplane --> cockpitiful
    # 8) lawyer airplane --> helicoptorney

    word1, word2 = raw_input("Seed words:  ").split(' ')

    n_neighbors = 300
    nearest_words1 = clean_words(word1, zip(*model.most_similar(positive=[word1], topn=n_neighbors))[0])
    nearest_words2 = clean_words(word2, zip(*model.most_similar(positive=[word2], topn=n_neighbors))[0])
    # nearest_words1 = clean_words(word1, [])
    # nearest_words2 = clean_words(word2, [])

    portmanteau_list = get_all_portmanteaus(nearest_words1, nearest_words2)
    portmanteau_list = clean_and_sort_results(portmanteau_list)

    # print the results
    print '\n'
    for portmanteau in portmanteau_list:
        print 'Phoneme:\t{}\nWord 1:\t\t{}\nWord 2:\t\t{}\nOverlap:\t{}\nDistance:\t{}\n'.format(portmanteau['portmanteau_phoneme'], portmanteau['first_word'], portmanteau['second_word'], portmanteau['overlap_size'], portmanteau['normed_phonetic_distance'])
