from global_constants import *
from helper_utils import *
import numpy as np
from nltk.corpus import wordnet as wn
from pun import Pun
from time import time

class Rhyme(Pun):
	min_overlap_vowel_phones = 1
	min_overlap_consonant_phones = 1
	min_overlap_phones = 2
	max_overlap_dist = 4

	def __init__(self,
							word1,
							word2,
							n_overlapping_vowel_phones,
							n_overlapping_consonant_phones,
							n_overlapping_phones,
							overlap_distance,
							overlap_grapheme_phoneme_prob,
							prob_given_tail_phoneme):
		self.word1 = word1
		self.word2 = word2
		self.n_overlapping_vowel_phones = n_overlapping_vowel_phones
		self.n_overlapping_consonant_phones = n_overlapping_consonant_phones
		self.n_overlapping_phones = n_overlapping_phones
		self.overlap_distance = overlap_distance
		self.overlap_grapheme_phoneme_prob = overlap_grapheme_phoneme_prob
		self.prob_given_tail_phoneme = prob_given_tail_phoneme

	@classmethod
	def get_pun(cls, word1, word2, subword_frequency):
		'''
		Attempts to create a Rhyme out of the two words
		If successful, returns the Rhyme
		If unnsuccessful, returns nil along with a descriptive error message

		We need the 'pronunciation_dictionary' in order to determine the probability of each grapheme
		'''

		# these are the default return values if no good overlaps are found
		rhyme, status, message = None, 1, 'no <=max_overlap_dist overlaps were found'

		min_word_len = min(len(word1.arpabet_phoneme), len(word2.arpabet_phoneme))
		# go large-to-small rather than small-to-large
		for overlap_len in range(min_word_len-1,0,-1):
			word1_arpabet_overlap = word1.arpabet_phoneme[-overlap_len:]
			word2_arpabet_overlap = word2.arpabet_phoneme[-overlap_len:]
			word1_arpabet_nonoverlap = word1.arpabet_phoneme[:-overlap_len]
			word2_arpabet_nonoverlap = word2.arpabet_phoneme[:-overlap_len]
			overlap_distance = cls.get_phoneme_distance(word1_arpabet_overlap, word2_arpabet_overlap)
			if overlap_distance <= cls.max_overlap_dist:
				# scrap the vectorizable phoneme mapping step, operate solely on the arpabet phoneme
				
				# Passes the initial distance test, now map the vectorizable_phoneme to the arpabet_phoneme, and check the remaining conditions
				
				# these redundant 'filter(str.isalpha, str(phone))' blocks are clunky, consider adding a function 'to_unstressed' or 'is_vowel'
				num_overlap_vowel_phones1 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_VOWELS else 0 for phone in word1_arpabet_overlap])
				num_overlap_consonant_phones1 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_CONSONANTS else 0 for phone in word1_arpabet_overlap])
				num_non_overlap_phones1 = len(word1_arpabet_nonoverlap)
				num_overlap_phones1 = len(word1_arpabet_overlap)
				num_non_overlap_phones2 = len(word2_arpabet_nonoverlap) # need to save this for later
				first_overlap_phone1 = filter(str.isalpha, str(word1_arpabet_overlap[0]))
				
				if num_overlap_vowel_phones1 < cls.min_overlap_vowel_phones:
					rhyme, status, message = None, 1, 'arpabet overlap does not have enough vowels'
					continue
				elif num_overlap_consonant_phones1 < cls.min_overlap_consonant_phones:
					rhyme, status, message = None, 1, 'arpabet overlap does not have enough consonants'
					continue
				elif num_overlap_phones1 < cls.min_overlap_phones:
					rhyme, status, message = None, 1, 'arpabet overlap does not have enough phones'
					continue

				word1_arpabet_overlap_start_idx, word1_arpabet_overlap_end_idx = len(word1.arpabet_phoneme) - overlap_len, len(word1.arpabet_phoneme) - 1
				word2_arpabet_overlap_start_idx, word2_arpabet_overlap_end_idx = len(word2.arpabet_phoneme) - overlap_len, len(word2.arpabet_phoneme) - 1
				try:
					word1_grapheme_overlap_start_idx, word1_grapheme_overlap_end_idx = word1.grapheme_to_arpabet_phoneme_alignment.subseq2_inds_to_subseq1_inds(word1_arpabet_overlap_start_idx, word1_arpabet_overlap_end_idx)
				except:
					rhyme, status, message = None, 1, 'word1 arpabet_phoneme could not be aligned with grapheme'
					continue
				
				try:
					word2_grapheme_overlap_start_idx, word2_grapheme_overlap_end_idx = word2.grapheme_to_arpabet_phoneme_alignment.subseq2_inds_to_subseq1_inds(word2_arpabet_overlap_start_idx, word2_arpabet_overlap_end_idx)
				except:
					rhyme, status, message = None, 1, 'word2 arpabet_phoneme could not be aligned with grapheme'
					continue

				# All alignments and min-char requirements have been met, so create the Portmanteau, and return

				# probability of a phoneme occurring with a particular occompanying grapheme is inversely proportional to the pun's quality; same is true for rhymes
				# basically, this is a roundabout way of identifying/discarding common prefixes/suffixes (commonly occurring graph/phone combinations at the start/end of words)

				word1_overlap_grapheme_phoneme_prob = cls.get_grapheme_phoneme_prob(word1.grapheme[word1_grapheme_overlap_start_idx:word1_grapheme_overlap_end_idx+1], tuple(word1_arpabet_overlap), subword_frequency)
				word2_overlap_grapheme_phoneme_prob = cls.get_grapheme_phoneme_prob(word2.grapheme[word2_grapheme_overlap_start_idx:word2_grapheme_overlap_end_idx+1], tuple(word2_arpabet_overlap), subword_frequency)
				overlap_grapheme_phoneme_prob = max(word1_overlap_grapheme_phoneme_prob, word2_overlap_grapheme_phoneme_prob)

				# Need to think more about this one...
				word1_prob_given_tail_phoneme = cls.get_prob_word_given_tail_subphoneme(tuple(word1_arpabet_overlap), subword_frequency)
				word2_prob_given_tail_phoneme = cls.get_prob_word_given_tail_subphoneme(tuple(word2_arpabet_overlap), subword_frequency)			
				prob_given_tail_phoneme = word1_prob_given_tail_phoneme * word2_prob_given_tail_phoneme

				# Use POS + grapheme_length ordering rules to decide which word to put first
				word1_ordered, word2_ordered = cls.get_word_ordering(word1, word2, num_non_overlap_phones1, num_non_overlap_phones2)

				rhyme = cls(
					word1_ordered,
					word2_ordered,
					# both overlap_phonemes1 and overlap_phonemes2 will have the same number of vowels and consonants, so can use either one
					num_overlap_vowel_phones1,
					num_overlap_consonant_phones1,
					num_overlap_vowel_phones1+num_overlap_consonant_phones1,
					overlap_distance,
					overlap_grapheme_phoneme_prob,
					prob_given_tail_phoneme
					)
				return rhyme, 0, 'rhyme found!'

		# failed to find any overlaps meeting the 'max_overlap_dist' criteria, so return with default error message
		return rhyme, status, message

	@staticmethod
	def get_word_ordering(word1, word2, num_non_overlap_phones1, num_non_overlap_phones2):
		'''
		How common is it for this particular graphic+phonetic element to occur at the start of end of a word?
		If it's extremely common, then it's probably a (garbage) common prefix/suffix
		'''

		# get top POS for word1 and word2 according to wordnet
		word1_synsets = wn.synsets(word1.grapheme)
		word2_synsets = wn.synsets(word2.grapheme)
		# both words exist in wordnet
		if word1_synsets and word2_synsets:
			pos1 = word1_synsets[0].pos()
			pos2 = word2_synsets[0].pos()
			if POS_ORDERING.get((pos1,pos2)) == 'keep':
				return word1, word2
			elif POS_ORDERING.get((pos1,pos2)) == 'flip':
				return word2, word1

		# if made it this far --> POS exists for both words, but isn't informative
		# therefore order the words such that the rhyming segments are as close together as possible
		if num_non_overlap_phones1 < num_non_overlap_phones2:
			return word2, word1
		else:
			return word1, word2

	def __repr__(self):
		return '''
		-------------------------------------------------------------------------------
		# Grapheme Pair: {} {}
		# Phoneme Pair: {} {}
		# Phoneme Distance: {}
		# Grapheme+Phoneme Probability: {}
		# Word Probability Given Phoneme + Length: {}
		# Overlapping Phones: {}
		# Overlapping Vowel Phones: {}
		# Overlapping Consonant Phones: {}
		-------------------------------------------------------------------------------
		'''.format(self.word1.grapheme,
			self.word2.grapheme,
			'-'.join(self.word1.arpabet_phoneme), # represented internally as a list, so collapse the list to a string
			'-'.join(self.word2.arpabet_phoneme), # represented internally as a list, so collapse the list to a string
			self.overlap_distance,
			round(self.overlap_grapheme_phoneme_prob, 5),
			round(self.prob_given_tail_phoneme, 5),
			self.n_overlapping_phones,
			self.n_overlapping_vowel_phones,
			self.n_overlapping_consonant_phones)

	def __str__(self):
		return '{} {}'.format(self.word1.grapheme, self.word2.grapheme)

	def ordering_criterion(self):
		'''Smaller values are "better" portmanteaus'''
		return (self.overlap_distance, -self.prob_given_tail_phoneme, -self.n_overlapping_phones)
		# return (-self.n_overlapping_phones, self.overlap_distance, self.overlap_grapheme_phoneme_prob)
		# return (-self.n_overlapping_phones, self.overlap_grapheme_phoneme_prob)
		# return (self.overlap_grapheme_phoneme_prob, -self.n_overlapping_phones, -self.n_overlapping_vowel_phones, self.overlap_distance)