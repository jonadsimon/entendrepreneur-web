from global_constants import *
import numpy as np
from models import SubgraphemeFrequency, SubphonemeFrequency

class Pun(object):
	'''
	---------------
	# DESCRIPTION #
	---------------
	Pun is the base class for Portmanteau and Rhyme

	-----------------
	# CLASS METHODS #
	-----------------
	get_pun : factory for generating Portmanteaus
	ordering_criterion : returns tuples indicating the quality of a given portmanteau
	'''
	# Class Constants
	MIN_OVERLAP_VOWEL_PHONES = 1
	MIN_OVERLAP_CONSONANT_PHONES = 1
	MIN_OVERLAP_PHONES = 2
	MAX_OVERLAP_DIST = 4
	MIN_NON_OVERLAP_PHONES = 1

	@classmethod
	def get_pun(cls, word1, word2):
		'''
		Implemented in the derived class
		'''
		pass

	@staticmethod
	def get_prob_word_given_subgrapheme(subgrapheme, side, subgrapheme_frequency_cache=None):
		'''
		Probability of a word given that it starts with/end with/contains a given grapheme
		'''
		if subgrapheme_frequency_cache: # cache is present
			subgrapheme_row = subgrapheme_frequency_cache.get(subgrapheme)
			if subgrapheme_row:
				return 1.0 / subgrapheme_row[side]
			else: # subgrapheme too long/rare to appear in table, assume freq=1
				return 1.0
		else: # cache is not present
			return 1.0 / SubgraphemeFrequency.get_subgrapheme_frequency(subgrapheme, side)

	@staticmethod
	def get_subphoneme_prob(subphoneme, side, subphoneme_frequency_cache=None):
		'''
		Probability of any word starting with/ending with/containing a given phoneme
		'''
		if subphoneme_frequency_cache: # cache is present
			subphoneme_row = subphoneme_frequency_cache.get(subphoneme)
			if subphoneme_row:
				return 1.0 * subphoneme_row[side] / VOCAB_SIZE
			else: # subphoneme too long/rare to appear in table, assume freq=1
				return 1.0 / VOCAB_SIZE
		else: # cache is not present
			return 1.0 * SubphonemeFrequency.get_subphoneme_frequency(subphoneme, side) / VOCAB_SIZE

	@staticmethod
	def get_phone_distance(phone1, phone2):
	    '''
		Compute phone-level distance:
		1) identical phones --> d=0
		2) unstressed phone vs primary stress phone --> d=1
		3) phones in near-miss consonant set --> d=2
		4) phones in near-miss vowel set --> d=4
		5) otherwise --> d=infty
	    '''
	    if phone1 == phone2:
	        return 0
	    elif filter(str.isalpha, phone1) == filter(str.isalpha, phone2):
	        # small penalty if identical BUT one has a primary stress and the other is nonstressed
	        if (phone1[-1], phone2[-1]) == ('0','1') or (phone1[-1], phone2[-1]) == ('1','0'):
	            return 1
	        # no penalty for secondary stress discrepancies
	        else:
	            return 0
	    elif (phone1,phone2) in NEAR_MISS_CONSONANTS or (phone2,phone1) in NEAR_MISS_CONSONANTS:
	        return 2
	    # make sure to strip off the stresses before checking for set inclusion
	    # if the vowels don't match then the stresses DEFINITELY need to match
	    elif ((phone1[:-1],phone2[:-1]) in NEAR_MISS_VOWELS or (phone2[:-1],phone1[:-1]) in NEAR_MISS_VOWELS) and phone1[-1] == phone2[-1]:
	        return 4
	    else:
	        return np.inf

	# Needs to be a classmethod rather than a static method so that it can call 'get_phone_distance'
	@classmethod
	def get_phoneme_distance(cls, phoneme1, phoneme2):
	    '''
	    Phoneme distance is the sum of pairwise phone distances
	    '''
	    return sum([cls.get_phone_distance(str(p1),str(p2)) for (p1,p2) in zip(phoneme1, phoneme2)])

	def ordering_criterion(self):
		'''
		Implemented in the derived class
		'''
		pass
