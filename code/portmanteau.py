from global_constants import *
import numpy as np
from pun import Pun

class Portmanteau(Pun):
	
	def __init__(self,
							word1,
							word2,
							grapheme_portmanteau1,
							grapheme_portmanteau2,
							arpabet_portmanteau1,
							arpabet_portmanteau2,
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
		self.arpabet_portmanteau1 = arpabet_portmanteau1
		self.arpabet_portmanteau2 = arpabet_portmanteau2
		self.reconstruction_proba1 = reconstruction_proba1
		self.reconstruction_proba2 = reconstruction_proba2		
		self.n_overlapping_vowel_phones = n_overlapping_vowel_phones
		self.n_overlapping_consonant_phones = n_overlapping_consonant_phones
		self.n_overlapping_phones = n_overlapping_phones
		self.overlap_distance = overlap_distance
		self.overlap_phoneme_prob = overlap_phoneme_prob

	@classmethod
	def get_pun(cls, word1, word2, subword_frequency):
		'''
		Attempts to create a Portmanteau out of the two words
		If successful, returns the Portmanteau
		If unnsuccessful, returns nil along with a descriptive error message

		We need the 'pronunciation_dictionary' in order to determine the probability of each grapheme
		'''

		# these are the default return values if no good overlaps are found
		portmanteau, status, message = None, 1, 'no <=max_overlap_dist overlaps were found'

		min_word_len = min(len(word1.arpabet_phoneme), len(word2.arpabet_phoneme))
		for overlap_len in range(1,min_word_len):
			word1_idx = len(word1.arpabet_phoneme) - overlap_len
			word2_idx = overlap_len
			word1_arpabet_overlap = word1.arpabet_phoneme[word1_idx:]
			word2_arpabet_overlap = word2.arpabet_phoneme[:word2_idx]
			word1_arpabet_nonoverlap = word1.arpabet_phoneme[:word1_idx]
			word2_arpabet_nonoverlap = word2.arpabet_phoneme[word2_idx:]
			overlap_distance = cls.get_phoneme_distance(word1_arpabet_overlap, word2_arpabet_overlap)
			if overlap_distance <= cls.max_overlap_dist:
				# scrap the vectorizable phoneme mapping step, operate solely on the arpabet phoneme

				# only possible to match vowels with vowels, and consonants with consonants, so don't bother checking both phonemes separately
				num_overlap_vowel_phones1 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_VOWELS else 0 for phone in word1_arpabet_overlap])
				num_overlap_consonant_phones1 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_CONSONANTS else 0 for phone in word1_arpabet_overlap])
				num_non_overlap_phones1 = len(word1_arpabet_nonoverlap)
				num_non_overlap_phones2 = len(word2_arpabet_nonoverlap)

				if num_overlap_vowel_phones1 < cls.min_overlap_vowel_phones:
					portmanteau, status, message = None, 1, 'arpabet overlap does not have enough vowels'
					continue
				elif num_overlap_consonant_phones1 < cls.min_overlap_consonant_phones:
					portmanteau, status, message = None, 1, 'arpabet overlap does not have enough consonants'
					continue
				elif num_non_overlap_phones1 < cls.min_non_overlap_phones:
					portmanteau, status, message = None, 1, 'word1 non-overlap does not have enough characters'
					continue
				elif num_non_overlap_phones2 < cls.min_non_overlap_phones:
					portmanteau, status, message = None, 1, 'word2 non-overlap does not have enough characters'
					continue

				# SUPER redundant, should be able to scrap this
				word1_arpabet_overlap_start_idx, word1_arpabet_overlap_end_idx = word1_idx, len(word1.arpabet_phoneme) - 1
				word2_arpabet_overlap_start_idx, word2_arpabet_overlap_end_idx = 0, word2_idx - 1
					
				try:
					word1_grapheme_overlap_start_idx, word1_grapheme_overlap_end_idx = word1.grapheme_to_arpabet_phoneme_alignment.subseq2_inds_to_subseq1_inds(word1_arpabet_overlap_start_idx, word1_arpabet_overlap_end_idx)
				except:
					portmanteau, status, message = None, 1, 'word1 arpabet_phoneme could not be aligned with grapheme'
					continue
				
				try:
					word2_grapheme_overlap_start_idx, word2_grapheme_overlap_end_idx = word2.grapheme_to_arpabet_phoneme_alignment.subseq2_inds_to_subseq1_inds(word2_arpabet_overlap_start_idx, word2_arpabet_overlap_end_idx)
				except:
					portmanteau, status, message = None, 1, 'word2 arpabet_phoneme could not be aligned with grapheme'
					continue

				# All alignments and min-char requirements have been met, so create the Portmanteau, and return

				# pick the graphemetric representation such that the constituent words are most easily reconstruble
				# i.e. use the innards of the word which is most easily predicted from its dangling phones

				word1_grapheme_nonoverlap = ''.join(word1.grapheme_to_arpabet_phoneme_alignment.subseq2_to_subseq1(0, word1_arpabet_overlap_start_idx-1))
				word2_grapheme_nonoverlap = ''.join(word2.grapheme_to_arpabet_phoneme_alignment.subseq2_to_subseq1(word2_arpabet_overlap_end_idx+1, len(word2.arpabet_phoneme)-1))

				word1_prob_given_dangling_graphs = cls.get_prob_word_given_subgrapheme(word1_grapheme_nonoverlap, 'head', subword_frequency)
				word2_prob_given_dangling_graphs = cls.get_prob_word_given_subgrapheme(word2_grapheme_nonoverlap, 'tail', subword_frequency)

				grapheme_portmanteau1 = word1.grapheme + word2_grapheme_nonoverlap
				grapheme_portmanteau2 = word1_grapheme_nonoverlap + word2.grapheme
				arpabet_portmanteau1 = word1.arpabet_phoneme + word2_arpabet_nonoverlap
				arpabet_portmanteau2 = word1_arpabet_nonoverlap + word2.arpabet_phoneme

				# If first word can be more easily reconstructed than the second, flip the ordering of the graphemes
				if word1_prob_given_dangling_graphs > word2_prob_given_dangling_graphs:
					grapheme_portmanteau1, grapheme_portmanteau2 = grapheme_portmanteau2, grapheme_portmanteau1
					arpabet_portmanteau1, arpabet_portmanteau2 = arpabet_portmanteau2, arpabet_portmanteau1

				word1_overlap_phoneme_prob = cls.get_tail_phoneme_prob(tuple(word1_arpabet_overlap), subword_frequency)
				word2_overlap_phoneme_prob = cls.get_head_phoneme_prob(tuple(word2_arpabet_overlap), subword_frequency)
				overlap_phoneme_prob = word1_overlap_phoneme_prob * word2_overlap_phoneme_prob

				portmanteau = cls(
					word1,
					word2,
					grapheme_portmanteau1,
					grapheme_portmanteau2,
					arpabet_portmanteau1,
					arpabet_portmanteau2,
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

		# failed to find any overlaps meeting the 'max_overlap_dist' criteria, so return with default error message
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
			'-'.join(self.arpabet_portmanteau1), # represented internally as a list, so collapse the list to a string
			'-'.join(self.arpabet_portmanteau2), # represented internally as a list, so collapse the list to a string
			self.n_overlapping_phones,
			self.overlap_distance,
			self.overlap_phoneme_prob
			)

	def __str__(self):
		return '{} ({}/{})'.format(self.grapheme_portmanteau1, self.word1.grapheme, self.word2.grapheme)

	def ordering_criterion(self):
		'''Smaller values are "better" portmanteaus'''
		return (self.overlap_distance, self.overlap_phoneme_prob)