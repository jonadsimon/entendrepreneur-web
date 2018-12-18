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
    phoneme, Array[String] : phoneme representation of the word
    grapheme_to_phoneme_alignment, SequenceAlignment : alignment between the grapheme and phoneme
    '''
    def __init__(self, grapheme, phoneme, grapheme_to_phoneme_alignment):
        self.grapheme = grapheme
        self.phoneme = phoneme
        self.grapheme_to_phoneme_alignment = grapheme_to_phoneme_alignment
