from global_constants import *
import numpy as np

class Pun(object):
	# Should set these using the *global* constants
	min_overlap_vowel_phones = 1
	min_overlap_consonant_phones = 1
	min_overlap_phones = 2
	max_overlap_dist = 4
	min_non_overlap_phones = 1

	# Want to only consider start of word, or want to also do middle of word?
	# Try *all* internal matches, so reason to restrict ourselves

	@classmethod
	def get_pun(cls, word1, word2, subword_frequency):
		pass

	@staticmethod
	def get_prob_word_given_subgrapheme(subgrapheme, side, subword_frequency):
		'''
		TOTALLY redundant with function in SubwordFrequency
		'''
		return 1.0 / subword_frequency.get_subgrapheme_frequency(subgrapheme, side)


	@staticmethod
	def get_prob_word_given_tail_subphoneme(subphoneme, subword_frequency):
		'''
		TOTALLY redundant with function in SubwordFrequency
		'''
		return 1.0 / subword_frequency.get_subphoneme_frequency(subphoneme, 'tail')

	@staticmethod
	def get_grapheme_phoneme_prob(subgrapheme, subphoneme, subword_frequency):
		'''
		TOTALLY redundant with function in SubwordFrequency
		'''
		return 1.0 * subword_frequency.get_subword_frequency(subgrapheme, subphoneme) / subword_frequency.vocab_size

	@staticmethod
	def get_head_grapheme_phoneme_prob(subgrapheme, subphoneme, subword_frequency):
		'''
		TOTALLY redundant with function in SubwordFrequency
		'''
		return 1.0 * subword_frequency.get_subword_frequency(subgrapheme, subphoneme, 'head') / subword_frequency.vocab_size

	@staticmethod
	def get_tail_grapheme_phoneme_prob(subgrapheme, subphoneme, subword_frequency):
		'''
		TOTALLY redundant with function in SubwordFrequency
		'''
		return 1.0 * subword_frequency.get_subword_frequency(subgrapheme, subphoneme, 'tail') / subword_frequency.vocab_size

	@staticmethod
	def get_head_phoneme_prob(subphoneme, subword_frequency):
		'''
		TOTALLY redundant with function in SubwordFrequency
		'''
		return 1.0 * subword_frequency.get_subphoneme_frequency(subphoneme, 'head') / subword_frequency.vocab_size

	@staticmethod
	def get_tail_phoneme_prob(subphoneme, subword_frequency):
		'''
		TOTALLY redundant with function in SubwordFrequency
		'''
		return 1.0 * subword_frequency.get_subphoneme_frequency(subphoneme, 'tail') / subword_frequency.vocab_size

	@staticmethod
	def get_phone_distance(phone1, phone2):
	    '''
	    identical pairs --> 0
	    unstressed / primary stress --> 1
	    near-match consonants --> 2
	    near-match vowels --> 4
	    non-matched  --> np.inf
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
	    # make sure to strip off the (possibly nonexistent) before checking for set inclusion
	    # if the vowels don't match then the stresses DEFINITELY need to match
	    # TODO: handle this via some sort of custom lookup function
	    elif ((phone1[:-1],phone2[:-1]) in NEAR_MISS_VOWELS or (phone2[:-1],phone1[:-1]) in NEAR_MISS_VOWELS) and phone1[-1] == phone2[-1]:
	        return 4
	    else:
	        return np.inf

	# needs to be a classmethod rather than a static method so that it can call 'get_phone_distance'
	@classmethod
	def get_phoneme_distance(cls, phoneme1, phoneme2):
	    '''
	    Don't use fancy hand-coded rules, keep the same distance logic, just swap in the new metric
	    '''
	    return sum([cls.get_phone_distance(str(p1),str(p2)) for (p1,p2) in zip(phoneme1, phoneme2)])

	def ordering_criterion(self):
		# possible can make this generally applicable
		pass
