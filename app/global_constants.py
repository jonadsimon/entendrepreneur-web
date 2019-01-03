########################
### GLOBAL CONSTANTS ###
########################

# Number of alignable graphemes in the CMU Pronouncing Dictionary
VOCAB_SIZE = 116002

ARPABET_VOWELS = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY', 'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX'])
ARPABET_CONSONANTS = set(['B', 'CH', 'D', 'DH', 'DX', 'EL', 'EM', 'EN', 'F', 'G', 'H', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'NX', 'P', 'Q', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH'])

ARPABET_DIPHTHONGS = set(['AW', 'AY', 'EY', 'OW', 'OY'])
ARPABET_RHOTICS = set(['ER'])

MAX_PORTMANTEAUS = 10
MAX_RHYMES = 10
MAX_NEIGHBORS = 100
MAX_VOCAB = 300000
# TEST_INPUT = 'rosemary marriott'
TEST_INPUT = 'master blaster'

REPO_HOME = '/Users/jonsimon/Code/pun_generator/entendrepreneur-web/'

POS_ORDERING = {
    ('n','v'): 'keep',
    ('v','n'): 'flip',
    ('a','n'): 'keep',
    ('n','a'): 'flip',
    ('s','n'): 'keep',
    ('n','s'): 'flip',
    ('n','r'): 'keep',
    ('r','n'): 'flip',
    ('a','v'): 'keep',
    ('v','a'): 'flip',
    ('r','a'): 'keep',
    ('a','r'): 'flip',
    ('s','v'): 'keep',
    ('v','s'): 'flip',
    ('r','s'): 'keep',
    ('s','r'): 'flip',
    ('r','v'): 'keep',
    ('v','r'): 'flip',
}

# distance-2 vowels that didn't make the cut: ('AA','AE'), ('AH','AO'), ('AO','UH')
NEAR_MISS_VOWELS = set([('AA','EH'),('AH','UH'),('EH','IH')])

# distance-2 consonants that didn't make the cut: ('S','TH'), ('DH','Z'), ('DH','R')
NEAR_MISS_CONSONANTS = set([('B','P'),('D','DH'),('D','T'),('DH','TH'),('F','V'), ('SH','ZH'),('CH','JH'),('S','Z')])
