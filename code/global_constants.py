from collections import defaultdict

########################
### GLOBAL CONSTANTS ###
########################

# Might be better off just hard-coding phonetically similar phones...
# in additional to total distance, add additional normalize cutoff
# i.e. <= 4 overall, and <2 normalized

ARPABET_VOWELS = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY', 'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX'])
ARPABET_CONSONANTS = set(['B', 'CH', 'D', 'DH', 'DX', 'EL', 'EM', 'EN', 'F', 'G', 'H', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'NX', 'P', 'Q', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH'])

ARPABET_DIPHTHONGS = set(['AW', 'AY', 'EY', 'OW', 'OY'])
ARPABET_RHOTICS = set(['ER'])

MAX_PORTMANTEAUS = 30
MAX_RHYMES = 30
MAX_NEIGHBORS = 100
MAX_VOCAB = 300000
FAST_VOCAB = 20000
# TEST_INPUT = 'labrador dormitory'
TEST_INPUT = 'glitter literati'
# TEST_INPUT = 'rosemary marriott'
# TEST_INPUT = 'sprocket locket' ### NEW DISTANCE MEASURE DOES NOT MATCH 'AH0'/'IH0'
# TEST_INPUT = 'master blaster'
# TEST_INPUT = 'sigh seismologist' ### CAN'T MATCH AFTER REMOVING PortmanteauInternal CLASS
# TEST_INPUT = 'programmer bro' ### CAN'T MATCH AFTER REMOVING PortmanteauInternal CLASS

REPO_HOME = '/Users/jonathansimon/code/what-do-you-call-a-bot/'

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
