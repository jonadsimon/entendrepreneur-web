import numpy as np

class Word(object):
    '''
	---------------
	# DESCRIPTION #
	---------------
	The Word class stores all of the grapheme/phoneme/alignment information for a given word.
    These Word objects are the values stored within a SequenceAlignment dictionary

	-------------------
	# CLASS VARIABLES #
	-------------------
	grapheme, String : grapheme representation of the word
    arpabet_phoneme, Array[String] : phoneme representation of the word
    grapheme_to_arpabet_phoneme_alignment, SequenceAlignment : alignment between the grapheme and phoneme

    TODO: Remove all Methods and Members relating to the now defunct phoneme vector representation
    '''
    def __init__(self, grapheme, arpabet_phoneme, vectorizable_phoneme, grapheme_to_arpabet_phoneme_alignment, arpabet_phoneme_to_vectorizable_phoneme_alignment):
        self.grapheme = grapheme
        self.arpabet_phoneme = arpabet_phoneme
        self.vectorizable_phoneme = vectorizable_phoneme # ***DEPRECATED, WILL BE REMOVED***
        self.grapheme_to_arpabet_phoneme_alignment = grapheme_to_arpabet_phoneme_alignment
        self.arpabet_phoneme_to_vectorizable_phoneme_alignment = arpabet_phoneme_to_vectorizable_phoneme_alignment # ***DEPRECATED, WILL BE REMOVED***

    def feature_vectors(self): # ***DEPRECATED, WILL BE REMOVED***
        '''
        Return a num_phones x num_features numpy array giving the vectorized phoneme
        Don't store this explicitly due to the ~5x increased memory cost, it's much easier to compute on the fly
        '''
        return np.array([p2v[phone] for phone in self.vectorizable_phoneme])
