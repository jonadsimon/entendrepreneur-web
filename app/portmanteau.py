from global_constants import *
import numpy as np
from pun import Pun
from time import time

class Portmanteau(Pun):
	'''
	---------------
	# DESCRIPTION #
	---------------
	The Portmanteau class is used to represent portmanteaus, which are constructed by
	combining two words in a phonetically natural way. See paper for details.

	-------------------
	# CLASS VARIABLES #
	-------------------
	word1, Word : first word appearing in the portmanteau
	word2, Word : second word appearing in the portmanteau
	grapheme_portmanteau1, String : primary grapheme represation of the portmanteau
	grapheme_portmanteau2, String : alternate grapheme represation of the portmanteau
	phoneme_portmanteau1, Array[String] : primary phonetic represation of the portmanteau
	phoneme_portmanteau2, Array[String] : alternate phonetic represation of the portmanteau
	reconstruction_proba1, Float : p(grapheme1, grapheme2 | grapheme_portmanteau1)
	reconstruction_proba2, Float : p(grapheme1, grapheme2 | grapheme_portmanteau2)
	n_overlapping_vowel_phones, Int : number of overlapping vowel phones in the portmanteau
	n_overlapping_consonant_phones, Int : number of overlapping consonant phones in the portmanteau
	n_overlapping_phones, Int : total number of overlapping phones in the portmanteau
	overlap_distance, Float : d(p_overlap, q_overlap) (see paper)
	overlap_phoneme_prob, Float : p(p_overlap, q_overlap) (see paper)

	-----------------
	# CLASS METHODS #
	-----------------
	get_pun : factory for generating Portmanteaus
	ordering_criterion : returns tuples indicating the quality of a given portmanteau
	'''
	# Class Constants
	ORDERING_CRITERION_CUTOFF = -7.5

	def __init__(self,
				word1,
				word2,
				grapheme_portmanteau1,
				grapheme_portmanteau2,
				phoneme_portmanteau1,
				phoneme_portmanteau2,
				reconstruction_proba1,
				reconstruction_proba2,
				n_overlapping_vowel_phones,
				n_overlapping_consonant_phones,
				n_overlapping_phones,
				overlap_distance,
				overlap_phoneme_prob
				):
		self.word1 = word1
		self.word2 = word2
		self.grapheme_portmanteau1 = grapheme_portmanteau1
		self.grapheme_portmanteau2 = grapheme_portmanteau2
		self.phoneme_portmanteau1 = phoneme_portmanteau1
		self.phoneme_portmanteau2 = phoneme_portmanteau2
		self.reconstruction_proba1 = reconstruction_proba1
		self.reconstruction_proba2 = reconstruction_proba2
		self.n_overlapping_vowel_phones = n_overlapping_vowel_phones
		self.n_overlapping_consonant_phones = n_overlapping_consonant_phones
		self.n_overlapping_phones = n_overlapping_phones
		self.overlap_distance = overlap_distance
		self.overlap_phoneme_prob = overlap_phoneme_prob

	@classmethod
	def get_pun(cls, word1, word2):
		'''
		---------------
		# DESCRIPTION #
		---------------
		Attempts to create a Portmanteau out of the two inputs words
		If successful, returns the Portmanteau
		If unnsuccessful, returns None along with a descriptive error message

		Two words can be combined into a portmanteau if:
		1) the phonemes at the tail of one word are in close phonetric proximity the phonemes at the head of the other word
		2) the phoneme overlap is at least 2 phones long
		3) the phoneme overlap contains at least 1 vowel
		4) the overlap does not comprise either of the entire words
		5) the overlapping phones can be brought into alignment with their corresponding graphemes

		See paper for details

		----------
		# INPUTS #
		----------
		word1, Word : first word in the (possible) portmanteau
		word2, Word : second word in the (possible) portmanteau

		-----------
		# OUTPUTS #
		-----------
		portmanteau, Portmanteau : either the generated portmanteau, or None if one is not found
		status, Int : 0 if a portmanteau is found, 1 otherwise
		message, String : message describing the success/failure status of the portmanteau construction
		'''

		# These are the default return values if no good overlaps are found
		portmanteau, status, message = None, 1, 'no <=MAX_OVERLAP_DIST overlaps were found'

		min_word_len = min(len(word1.phoneme), len(word2.phoneme))
		for overlap_len in range(1,min_word_len):
			word1_idx = len(word1.phoneme) - overlap_len
			word2_idx = overlap_len
			word1_phoneme_overlap = word1.phoneme[word1_idx:]
			word2_phoneme_overlap = word2.phoneme[:word2_idx]
			word1_phoneme_nonoverlap = word1.phoneme[:word1_idx]
			word2_phoneme_nonoverlap = word2.phoneme[word2_idx:]
			overlap_distance = cls.get_phoneme_distance(word1_phoneme_overlap, word2_phoneme_overlap)
			if overlap_distance <= cls.MAX_OVERLAP_DIST:
				# It's only possible to match vowels with vowels, and consonants with consonants, so only need to run the check on one of the phonemes
				num_overlap_vowel_phones1 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_VOWELS else 0 for phone in word1_phoneme_overlap])
				num_overlap_consonant_phones1 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_CONSONANTS else 0 for phone in word1_phoneme_overlap])
				num_non_overlap_phones1 = len(word1_phoneme_nonoverlap)
				num_non_overlap_phones2 = len(word2_phoneme_nonoverlap)

				# Verify the the overlapping/nonoverlapping phones satisfy the desired constraints on e.g. length
				if num_overlap_vowel_phones1 < cls.MIN_OVERLAP_VOWEL_PHONES:
					portmanteau, status, message = None, 1, 'phoneme overlap does not have enough vowels'
					continue
				elif num_overlap_consonant_phones1 < cls.MIN_OVERLAP_CONSONANT_PHONES:
					portmanteau, status, message = None, 1, 'phoneme overlap does not have enough consonants'
					continue
				elif num_non_overlap_phones1 < cls.MIN_NON_OVERLAP_PHONES:
					portmanteau, status, message = None, 1, 'word1 non-overlap does not have enough characters'
					continue
				elif num_non_overlap_phones2 < cls.MIN_NON_OVERLAP_PHONES:
					portmanteau, status, message = None, 1, 'word2 non-overlap does not have enough characters'
					continue

				# Highly redundant, consider scrapping
				word1_phoneme_overlap_start_idx, word1_phoneme_overlap_end_idx = word1_idx, len(word1.phoneme) - 1
				word2_phoneme_overlap_start_idx, word2_phoneme_overlap_end_idx = 0, word2_idx - 1

				# The phonemes contain a viable overlap, but the overlap cannot be brought into alignment with the first grapheme
				try:
					word1_grapheme_overlap_start_idx, word1_grapheme_overlap_end_idx = word1.get_subgrapheme_from_subphoneme_inds(word1_phoneme_overlap_start_idx, word1_phoneme_overlap_end_idx, return_inds=True)
				except:
					portmanteau, status, message = None, 1, 'word1 phoneme could not be aligned with grapheme'
					continue

				# The phonemes contain a viable overlap, but the overlap cannot be brought into alignment with the second grapheme
				try:
					word2_grapheme_overlap_start_idx, word2_grapheme_overlap_end_idx = word2.get_subgrapheme_from_subphoneme_inds(word2_phoneme_overlap_start_idx, word2_phoneme_overlap_end_idx, return_inds=True)
				except:
					portmanteau, status, message = None, 1, 'word2 phoneme could not be aligned with grapheme'
					continue

				# All alignments and min-char requirements have been met, so create the Portmanteau, and return it

				# Select the graphemetric representation such that the constituent words are most easily reconstruble
				# i.e. choose the grapheme_portmanteau which maximizes p(grapheme1, grapheme2 | grapheme_portmanteau)
				# See paper for details

				word1_grapheme_nonoverlap = ''.join(word1.get_subgrapheme_from_subphoneme_inds(0, word1_phoneme_overlap_start_idx-1, return_inds=False))
				word2_grapheme_nonoverlap = ''.join(word2.get_subgrapheme_from_subphoneme_inds(word2_phoneme_overlap_end_idx+1, len(word2.phoneme)-1, return_inds=False))

				start = time()
				word1_prob_given_dangling_graphs = cls.get_prob_word_given_subgrapheme(word1_grapheme_nonoverlap, 'head')
				word2_prob_given_dangling_graphs = cls.get_prob_word_given_subgrapheme(word2_grapheme_nonoverlap, 'tail')
				print "Subgrapheme proba (2x): {:.2f} seconds".format(time()-start)

				grapheme_portmanteau1 = word1.grapheme + word2_grapheme_nonoverlap
				grapheme_portmanteau2 = word1_grapheme_nonoverlap + word2.grapheme
				phoneme_portmanteau1 = word1.phoneme + word2_phoneme_nonoverlap
				phoneme_portmanteau2 = word1_phoneme_nonoverlap + word2.phoneme

				# If first word can be more easily reconstructed than the second, flip the ordering of the graphemes
				# This ensures that the grapheme_portmanteau maximizing p(grapheme1, grapheme2 | grapheme_portmanteau)
				# will be stored in the variable 'grapheme_portmanteau1'
				if word1_prob_given_dangling_graphs > word2_prob_given_dangling_graphs:
					grapheme_portmanteau1, grapheme_portmanteau2 = grapheme_portmanteau2, grapheme_portmanteau1
					phoneme_portmanteau1, phoneme_portmanteau2 = phoneme_portmanteau2, phoneme_portmanteau1

				# Compute p(p_overlap, q_overlap) (see paper)
				start = time()
				word1_overlap_phoneme_prob = cls.get_subphoneme_prob(tuple(word1_phoneme_overlap), 'tail')
				word2_overlap_phoneme_prob = cls.get_subphoneme_prob(tuple(word2_phoneme_overlap), 'head')
				print "Subphoneme proba (2x): {:.2f} seconds".format(time()-start)
				overlap_phoneme_prob = word1_overlap_phoneme_prob * word2_overlap_phoneme_prob

				# Instantiate the constructed portmanteau, and return it
				portmanteau = cls(
					word1,
					word2,
					grapheme_portmanteau1,
					grapheme_portmanteau2,
					phoneme_portmanteau1,
					phoneme_portmanteau2,
					word1_prob_given_dangling_graphs,
					word2_prob_given_dangling_graphs,
					# both overlap_phonemes1 and overlap_phonemes2 will have the same number of vowels and consonants, so can use either one
					num_overlap_vowel_phones1,
					num_overlap_consonant_phones1,
					num_overlap_vowel_phones1+num_overlap_consonant_phones1,
					overlap_distance,
					overlap_phoneme_prob
					)
				return portmanteau, 0, 'portmanteau found!'

		# Failed to find any overlaps meeting the 'MAX_OVERLAP_DIST' criteria, so return with the default error message
		return portmanteau, status, message

	def __repr__(self):
		return '''
		-------------------------------------------------------------------------------
		# Word Combination: {} + {}
		# Grapheme Portmanteau: {} ({})
		# Phoneme Portmanteau: {} ({})
		# Overlapping Phones: {}
		# Phoneme Distance: {}
		# Phoneme Probability: {:.2e}
		-------------------------------------------------------------------------------
		'''.format(self.word1.grapheme,
			self.word2.grapheme,
			self.grapheme_portmanteau1,
			self.grapheme_portmanteau2,
			'-'.join(self.phoneme_portmanteau1), # represented internally as a list, so collapse the list to a string
			'-'.join(self.phoneme_portmanteau2), # represented internally as a list, so collapse the list to a string
			self.n_overlapping_phones,
			self.overlap_distance,
			self.overlap_phoneme_prob
			)

	def __str__(self):
		return '{} ({}/{})'.format(self.grapheme_portmanteau1, self.word1.grapheme, self.word2.grapheme)

	def serialize(self):
		'''
		"&#xb7;" is the HTML symbol for "middot"
		http://www.fileformat.info/info/unicode/char/b7/index.htm
		'''
		return {
			'grapheme_portmanteau': self.grapheme_portmanteau1,
			'grapheme1': self.word1.grapheme,
			'grapheme2': self.word2.grapheme,
			'phoneme_portmanteau': '&#xb7;'.join(map(Portmanteau.subscript_phone_stress, self.phoneme_portmanteau1)),
			'phoneme1': '&#xb7;'.join(map(Portmanteau.subscript_phone_stress, self.word1.phoneme)),
			'phoneme2': '&#xb7;'.join(map(Portmanteau.subscript_phone_stress, self.word2.phoneme)),
			'phonetic_distance': '{:d}'.format(self.overlap_distance),
			'phonetic_probability': '{:.2e}'.format(self.overlap_phoneme_prob)
			}

	def ordering_criterion(self):
		'''
		Return a scalar used for ordering the Portmanteaus in terms of quality
		Smaller values correspond to "better" portmanteaus
		'''
		overlap_distance_coef, overlap_phoneme_prob_coef = 0.62, 0.79
		return overlap_distance_coef * self.overlap_distance + overlap_phoneme_prob_coef * np.log(self.overlap_phoneme_prob)
