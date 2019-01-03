# CMU Pronouncing Dictionary is already downcased, so just need to ensure that
# we only allow graphemes comprised of:
# 1) alpha characters
# 2) hyphens '-'
# 3) underscores '_'
#
# Takes ~10min to run m2m-aligner on the resulting preprocessed corpus
# Run m2m-aligner in command line as:
# >  PATH/TO/M2M_ALIGNER/m2m-aligner --delX --maxX 2 --maxY 2 -i data/m2m_preprocessed_cmudict.txt


import nltk.corpus import cmudict
import re

cmu_dict = cmudict.dict()
pattern = re.compile("^([a-z_\-]+)+$")

with open('../data/g2p_alignment/m2m_preprocessed_cmudict.txt', 'w') as outfile:
    for grapheme, phoneme_list in cmu_dict.iteritems():
        if pattern.match(grapheme):
            # Strip off stresses from phonemes, these confuse the aligner more than they help
            clean_phoneme = [filter(str.isalpha, str(phone)) for phone in phoneme_list[0]]
            outfile.write(' '.join(grapheme) + '\t' + ' '.join(clean_phoneme) + '\n')
