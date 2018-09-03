from global_constants import PHONOLOGICAL_PHONE_TO_PHONOLOGICAL_FEATURE_DICT as p2v
import numpy as np

class Word(object):
    '''
    A word is a grapheme-phoneme pair
      grapheme - str
      arpabet_phoneme  - list(str)
      vectorizable_phoneme  - list(str)
      grapheme_to_arpabet_phoneme_alignment - SequenceAlignment
      arpabet_phoneme_to_vectorizable_phoneme_alignment - SequenceAlignment

      phonetic_features - numpy_array(int) [n_pronounciation_phonemes x n_features]

      phonemes are assumed to be stress-less

      TODO: add __repr__ function, and rebuild the Pronounciation dictionary
    '''
    def __init__(self, grapheme, arpabet_phoneme, vectorizable_phoneme, grapheme_to_arpabet_phoneme_alignment, arpabet_phoneme_to_vectorizable_phoneme_alignment):
        self.grapheme = grapheme
        self.arpabet_phoneme = arpabet_phoneme
        self.vectorizable_phoneme = vectorizable_phoneme
        self.grapheme_to_arpabet_phoneme_alignment = grapheme_to_arpabet_phoneme_alignment
        self.arpabet_phoneme_to_vectorizable_phoneme_alignment = arpabet_phoneme_to_vectorizable_phoneme_alignment

    def feature_vectors(self):
        '''
        Return a num_phones x num_features numpy array giving the vectorized phoneme
        Don't store this explicitly due to the ~5x increased memory cost, it's much easier to compute on the fly
        '''
        return np.array([p2v[phone] for phone in self.vectorizable_phoneme])