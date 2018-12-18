from global_constants import *
from helper_utils import *
import numpy as np
from nltk.corpus import wordnet as wn
from pun import Pun
from time import time

class Rhyme(Pun):
	'''
	---------------
	# DESCRIPTION #
	---------------
	The Rhyme class is used to represent rhymes, which are comprised of two
	two words with matching tail phonemes.

	-------------------
	# CLASS VARIABLES #
	-------------------
	word1, Word : first word appearing in the rhyme
	word2, Word : second word appearing in the rhyme
	n_overlapping_vowel_phones, Int : number of overlapping vowel phones in the rhyme
	n_overlapping_consonant_phones, Int : number of overlapping consonant phones in the rhyme
	n_overlapping_phones, Int : total number of overlapping phones in the rhyme
	overlap_distance, Float : d(p_overlap, q_overlap) (see paper)
	overlap_phoneme_prob, Float : p(p_overlap, q_overlap) (see paper)

	-----------------
	# CLASS METHODS #
	-----------------
	get_pun : factory for generating Rhymes
	get_word_ordering : returns pair of rhyming words in the order they should be displayed
	ordering_criterion : returns tuples indicating the quality of a given rhyme
	'''
	def __init__(self,
				word1,
				word2,
				n_overlapping_vowel_phones,
				n_overlapping_consonant_phones,
				n_overlapping_phones,
				overlap_distance,
				overlap_phoneme_prob):
		self.word1 = word1
		self.word2 = word2
		self.n_overlapping_vowel_phones = n_overlapping_vowel_phones
		self.n_overlapping_consonant_phones = n_overlapping_consonant_phones
		self.n_overlapping_phones = n_overlapping_phones
		self.overlap_distance = overlap_distance
		self.overlap_phoneme_prob = overlap_phoneme_prob

	@classmethod
	def get_pun(cls, word1, word2, subword_frequency):
		'''
		---------------
		# DESCRIPTION #
		---------------
		Attempts to create a Rhyme out of the two inputs words
		If successful, returns the Rhyme
		If unnsuccessful, returns None along with a descriptive error message

		Two words can be combined into a rhyme if:
		1) the phonemes at the tail of one word are in close phonetric proximity the phonemes at the tail of the other word
		2) the phoneme overlap is at least 2 phones long
		3) the phoneme overlap begins with a vowel

		See paper for details

		----------
		# INPUTS #
		----------
		word1, Word : first word in the (possible) rhyme
		word2, Word : second word in the (possible) rhyme
		subword_frequency, SubwordFrequency : lookup table of subgrapheme/subphoneme frequencies

		-----------
		# OUTPUTS #
		-----------
		rhyme, Rhyme : either the generated rhyme, or None if one is not found
		status, Int : 0 if a rhyme is found, 1 otherwise
		message, String : message describing the success/failure status of the rhyme construction
		'''

		# These are the default return values if no good overlaps are found
		rhyme, status, message = None, 1, 'no <=MAX_OVERLAP_DIST overlaps were found'

		min_word_len = min(len(word1.phoneme), len(word2.phoneme))
		# Iterate in reverse, so that the largest phoneme overlap is identified
		for overlap_len in range(min_word_len-1,0,-1):
			word1_phoneme_overlap = word1.phoneme[-overlap_len:]
			word2_phoneme_overlap = word2.phoneme[-overlap_len:]
			word1_phoneme_nonoverlap = word1.phoneme[:-overlap_len]
			word2_phoneme_nonoverlap = word2.phoneme[:-overlap_len]
			overlap_distance = cls.get_phoneme_distance(word1_phoneme_overlap, word2_phoneme_overlap)
			if overlap_distance <= cls.MAX_OVERLAP_DIST:
				# It's only possible to match vowels with vowels, and consonants with consonants, so only need to run the check on one of the phonemes
				num_overlap_vowel_phones1 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_VOWELS else 0 for phone in word1_phoneme_overlap])
				num_overlap_consonant_phones1 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_CONSONANTS else 0 for phone in word1_phoneme_overlap])
				num_non_overlap_phones1 = len(word1_phoneme_nonoverlap)
				num_overlap_phones1 = len(word1_phoneme_overlap)
				num_non_overlap_phones2 = len(word2_phoneme_nonoverlap) # need to save this for later
				first_overlap_phone1 = filter(str.isalpha, str(word1_phoneme_overlap[0]))

				# Verify the the overlapping/nonoverlapping phones satisfy the desired constraints on e.g. length
				if num_overlap_vowel_phones1 < cls.MIN_OVERLAP_VOWEL_PHONES:
					rhyme, status, message = None, 1, 'phoneme overlap does not have enough vowels'
					continue
				elif num_overlap_consonant_phones1 < cls.MIN_OVERLAP_CONSONANT_PHONES:
					rhyme, status, message = None, 1, 'phoneme overlap does not have enough consonants'
					continue
				elif num_overlap_phones1 < cls.MIN_OVERLAP_PHONES:
					rhyme, status, message = None, 1, 'phoneme overlap does not have enough phones'
					continue
				elif first_overlap_phone1 not in ARPABET_VOWELS:
					rhyme, status, message = None, 1, 'phoneme overlap does not start with a vowel phone'
					continue

				# Highly redundant, consider scrapping
				word1_phoneme_overlap_start_idx, word1_phoneme_overlap_end_idx = len(word1.phoneme) - overlap_len, len(word1.phoneme) - 1
				word2_phoneme_overlap_start_idx, word2_phoneme_overlap_end_idx = len(word2.phoneme) - overlap_len, len(word2.phoneme) - 1

				# The phonemes contain a viable overlap, but the overlap cannot be brought into alignment with the first grapheme
				try:
					word1_grapheme_overlap_start_idx, word1_grapheme_overlap_end_idx = word1.grapheme_to_phoneme_alignment.subseq2_inds_to_subseq1_inds(word1_phoneme_overlap_start_idx, word1_phoneme_overlap_end_idx)
				except:
					rhyme, status, message = None, 1, 'word1 phoneme could not be aligned with grapheme'
					continue

				# The phonemes contain a viable overlap, but the overlap cannot be brought into alignment with the second grapheme
				try:
					word2_grapheme_overlap_start_idx, word2_grapheme_overlap_end_idx = word2.grapheme_to_phoneme_alignment.subseq2_inds_to_subseq1_inds(word2_phoneme_overlap_start_idx, word2_phoneme_overlap_end_idx)
				except:
					rhyme, status, message = None, 1, 'word2 phoneme could not be aligned with grapheme'
					continue

				# All alignments and min-char requirements have been met, so create the Rhyme, and return it

				# Compute p(p_overlap, q_overlap) (see paper)
				word1_tail_phoneme_prob = cls.get_subphoneme_prob(tuple(word1_phoneme_overlap), 'tail', subword_frequency)
				word2_tail_phoneme_prob = cls.get_subphoneme_prob(tuple(word2_phoneme_overlap), 'tail', subword_frequency)
				overlap_phoneme_prob = word1_tail_phoneme_prob * word2_tail_phoneme_prob

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
					overlap_phoneme_prob
					)
				return rhyme, 0, 'rhyme found!'

		# failed to find any overlaps meeting the 'MAX_OVERLAP_DIST' criteria, so return with default error message
		return rhyme, status, message

	@staticmethod
	def get_word_ordering(word1, word2, num_non_overlap_phones1, num_non_overlap_phones2):
		'''
		---------------
		# DESCRIPTION #
		---------------
		Given 2 words comprising a rhyme, order them such that their parts-of-speech follow naturally,
		i.e. adjective precedes noun, noun precedes verb, adverb vs verb, adverb precedes adjective

		If the two words' parts of speech are the same, or cannot be identified, then order the words
		such that the the phoneme overlaps are as close together as possible. Typically this will involve
		placing the longer word before the shorter word.

		----------
		# INPUTS #
		----------
		word1, Word : first word in the rhyme
		word2, Word : second word in the rhyme
		num_non_overlap_phones1, Int : number of phones at the head of phoneme1 not contained in the overlap
		num_non_overlap_phones2, Int : number of phones at the head of phoneme2 not contained in the overlap

		-----------
		# OUTPUTS #
		-----------
		word1, Word : first word in the rhyme, ordered according to POS or phonetic proximity
		word2, Word : second word in the rhyme, ordered according to POS or phonetic proximity
		'''

		# Get primary POS for word1 and word2 according to WordNet
		word1_synsets = wn.synsets(word1.grapheme)
		word2_synsets = wn.synsets(word2.grapheme)

		# If both words exist in WordNet, attempt to order them according to the POS_ORDERING
		# rules laid out in global_constants.py
		if word1_synsets and word2_synsets:
			pos1 = word1_synsets[0].pos()
			pos2 = word2_synsets[0].pos()
			if POS_ORDERING.get((pos1,pos2)) == 'keep':
				return word1, word2
			elif POS_ORDERING.get((pos1,pos2)) == 'flip':
				return word2, word1

		# Either the POS for one (or both) of the words doesn't exist, or else they have the same POS
		# In this case, order the words to put the phoneme overlaps in as close proximity as possible
		# e.g. 'radar car' instead of 'car radar' because the first places the 'ar' sounds closer together
		if num_non_overlap_phones1 < num_non_overlap_phones2:
			return word2, word1
		else:
			return word1, word2

	def __repr__(self):
		return '''
		-------------------------------------------------------------------------------
		# Grapheme Pair: {} {}
		# Phoneme Pair: {} {}
		# Overlapping Phones: {}
		# Phoneme Distance: {}
		# Phoneme Probability: {:.2e}
		-------------------------------------------------------------------------------
		'''.format(self.word1.grapheme,
			self.word2.grapheme,
			'-'.join(self.word1.phoneme), # represented internally as a list, so collapse the list to a string
			'-'.join(self.word2.phoneme), # represented internally as a list, so collapse the list to a string
			self.n_overlapping_phones,
			self.overlap_distance,
			self.overlap_phoneme_prob
			)

	def __str__(self):
		return '{} {}'.format(self.word1.grapheme, self.word2.grapheme)

	def ordering_criterion(self):
		'''
		Return a tuple used for ordering the Rhymes in terms of quality
		Smaller values correspond to "better" rhymes
		'''
		return (self.overlap_distance, self.overlap_phoneme_prob)
