from global_constants import *
import numpy as np
import pdb

class Portmanteau(object):
	# Should set these using the *global* constants
	min_overlap_vowel_phones = 1
	min_overlap_consonant_phones = 1
	min_non_overlap_phones = 1
	max_overlap_dist = 4
	
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
							overlap_grapheme_phoneme_prob):
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
		self.overlap_grapheme_phoneme_prob = overlap_grapheme_phoneme_prob

	@classmethod
	def get_portmanteau(cls, word1, word2, pronunciation_dictionary):
		'''
		Attempts to create a Portmanteau out of the two words
		If successful, returns the Portmanteau
		If unnsuccessful, returns nil along with a descriptive error message

		We need the 'pronunciation_dictionary' in order to determine the probability of each grapheme
		'''

		# these are the default return values if no good overlaps are found
		portmanteau, status, message = None, 1, 'no <=max_overlap_dist overlaps were found'

		min_word_len = min(len(word1.vectorizable_phoneme), len(word2.vectorizable_phoneme))
		for overlap_len in range(1,min_word_len):
			# if overlap_len == 4:
			# 	pdb.set_trace()
			word1_idx = len(word1.vectorizable_phoneme) - overlap_len
			word2_idx = overlap_len
			word1_vector_overlap = word1.feature_vectors()[word1_idx:]
			word2_vector_overlap = word2.feature_vectors()[:word2_idx]
			overlap_distance = abs(word1_vector_overlap - word2_vector_overlap).sum()
			if overlap_distance <= cls.max_overlap_dist:
				# Passes the initial distance test, now map the vectorizable_phoneme to the arpabet_phoneme, and check the remaining conditions
				try:
					word1_arpabet_overlap = word1.arpabet_phoneme_to_vectorizable_phoneme_alignment.subseq2_to_subseq1(word1_idx, len(word1.vectorizable_phoneme)-1)
					word1_arpabet_nonoverlap = word1.arpabet_phoneme_to_vectorizable_phoneme_alignment.subseq2_to_subseq1(0, word1_idx-1)
				except:
					portmanteau, status, message = None, 1, 'word1 vectorizable_phoneme could not be aligned with arpabet_phoneme'
					continue
				else:
					# these redundant 'filter(str.isalpha, str(phone))' blocks are clunky, consider adding a function 'to_unstressed' or 'is_vowel'
					num_overlap_vowel_phones1 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_VOWELS else 0 for phone in word1_arpabet_overlap])
					num_overlap_consonant_phones1 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_CONSONANTS else 0 for phone in word1_arpabet_overlap])
					num_non_overlap_phones1 = len(word1_arpabet_nonoverlap)
					
					if num_overlap_vowel_phones1 < cls.min_overlap_vowel_phones:
						portmanteau, status, message = None, 1, 'word1 overlap does not have enough vowels'
						continue
					elif num_overlap_consonant_phones1 < cls.min_overlap_consonant_phones:
						portmanteau, status, message = None, 1, 'word1 overlap does not have enough consonants'
						continue
					elif num_non_overlap_phones1 < cls.min_non_overlap_phones:
						portmanteau, status, message = None, 1, 'word1 non-overlap does not have enough characters'
						continue

				try:
					word2_arpabet_overlap = word2.arpabet_phoneme_to_vectorizable_phoneme_alignment.subseq2_to_subseq1(0, word2_idx-1)
					word2_arpabet_nonoverlap = word2.arpabet_phoneme_to_vectorizable_phoneme_alignment.subseq2_to_subseq1(word2_idx, len(word2.vectorizable_phoneme)-1)
				except:
					portmanteau, status, message = None, 1, 'word2 vectorizable_phoneme could not be aligned with arpabet_phoneme'
					continue
				else:
					num_overlap_vowel_phones2 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_VOWELS else 0 for phone in word2_arpabet_overlap])
					num_overlap_consonant_phones2 = sum([1 if filter(str.isalpha, str(phone)) in ARPABET_CONSONANTS else 0 for phone in word2_arpabet_overlap])
					num_non_overlap_phones2 = len(word2_arpabet_nonoverlap)

					if num_overlap_vowel_phones2 < cls.min_overlap_vowel_phones:
						portmanteau, status, message = None, 1, 'word2 overlap does not have enough vowels'
						continue
					elif num_overlap_consonant_phones2 < cls.min_overlap_consonant_phones:
						portmanteau, status, message = None, 1, 'word2 overlap does not have enough consonants'
						continue
					elif num_non_overlap_phones2 < cls.min_non_overlap_phones:
						portmanteau, status, message = None, 1, 'word2 non-overlap does not have enough characters'
						continue

				# if overlap_len == 4:
				# 	pdb.set_trace()

				# vectorizable_phoneme-to-arpabet_phoneme alignments worked *and* all arpabet constraints are met
				# possible for one grapheme alignment to work, but not the other, HOWEVER enforce that they BOTH work for ease of future logic
				# (this will throw away some otherwise fine portmanteaus, so return to handle this edge case later)
				try:
					word1_arpabet_overlap_start_idx, word1_arpabet_overlap_end_idx = word1.arpabet_phoneme_to_vectorizable_phoneme_alignment.subseq2_inds_to_subseq1_inds(word1_idx, len(word1.vectorizable_phoneme)-1)
					word1_grapheme_overlap_start_idx, word1_grapheme_overlap_end_idx = word1.grapheme_to_arpabet_phoneme_alignment.subseq2_inds_to_subseq1_inds(word1_arpabet_overlap_start_idx, word1_arpabet_overlap_end_idx)
				except:
					portmanteau, status, message = None, 1, 'word1 arpabet_phoneme could not be aligned with grapheme'
					continue
				
				try:
					word2_arpabet_overlap_start_idx, word2_arpabet_overlap_end_idx = word2.arpabet_phoneme_to_vectorizable_phoneme_alignment.subseq2_inds_to_subseq1_inds(0, word2_idx-1)
					word2_grapheme_overlap_start_idx, word2_grapheme_overlap_end_idx = word2.grapheme_to_arpabet_phoneme_alignment.subseq2_inds_to_subseq1_inds(word2_arpabet_overlap_start_idx, word2_arpabet_overlap_end_idx)
				except:
					portmanteau, status, message = None, 1, 'word2 arpabet_phoneme could not be aligned with grapheme'
					continue

				# All alignments and min-char requirements have been met, so create the Portmanteau, and return

				# pick the graphemetric representation such that the constituent words are most easily reconstruble
				# i.e. use the innards of the word which is most easily predicted from its dangling phones

				# TWO GLARING ISSUES: FIGURE OUT HOW TO FIX THEM
				# Grapheme Portmanteau: labradordormitory (labradorrmitory)
				# Phoneme Portmanteau: L-AE1-B-R-D-AO1-R-M-AH0-T-AO2-R-IY0 (L-AE1-B-R-AH0-D-AO2-R-M-AH0-T-AO2-R-IY0)

				# may be rife with OBO errors, do another round to double-check afterwards

				# manual overlap calculation performs poorly on silent letters, so use the explicit calculation instead

				# word1_grapheme_nonoverlap_start_idx, word1_grapheme_nonoverlap_end_idx = 0, word1_grapheme_overlap_start_idx - 1
				# word2_grapheme_nonoverlap_start_idx, word2_grapheme_nonoverlap_end_idx = word2_grapheme_overlap_end_idx + 1, len(word2.grapheme) - 1

				# don't need the indices, just need the strings

				word1_grapheme_nonoverlap = ''.join(word1.grapheme_to_arpabet_phoneme_alignment.subseq2_to_subseq1(0, word1_arpabet_overlap_start_idx-1))
				word2_grapheme_nonoverlap = ''.join(word2.grapheme_to_arpabet_phoneme_alignment.subseq2_to_subseq1(word2_arpabet_overlap_end_idx+1, len(word2.arpabet_phoneme)-1))

				# word1_prob_given_dangling_graphs = cls.get_word_prob_from_subgraph(word1.grapheme, word1_grapheme_nonoverlap_start_idx, word1_grapheme_nonoverlap_end_idx, pronunciation_dictionary)
				# word2_prob_given_dangling_graphs = cls.get_word_prob_from_subgraph(word2.grapheme, word2_grapheme_nonoverlap_start_idx, word2_grapheme_nonoverlap_end_idx, pronunciation_dictionary)

				word1_prob_given_dangling_graphs = cls.get_prob_word_given_subgrapheme(word1_grapheme_nonoverlap, 'head', pronunciation_dictionary)
				word2_prob_given_dangling_graphs = cls.get_prob_word_given_subgrapheme(word2_grapheme_nonoverlap, 'tail', pronunciation_dictionary)

				# should do compute these in this same manual way above as well
				# word1_arpabet_nonoverlap_end_idx = word1_arpabet_overlap_start_idx - 1
				# word2_arpabet_nonoverlap_start_idx = word2_arpabet_overlap_end_idx + 1

				grapheme_portmanteau1 = word1.grapheme + word2_grapheme_nonoverlap
				grapheme_portmanteau2 = word1_grapheme_nonoverlap + word2.grapheme
				arpabet_portmanteau1 = word1.arpabet_phoneme + word2_arpabet_nonoverlap
				arpabet_portmanteau2 = word1_arpabet_nonoverlap + word2.arpabet_phoneme

				# grapheme_portmanteau1 = word1.grapheme + word2.grapheme[word2_grapheme_nonoverlap_start_idx:]
				# grapheme_portmanteau2 = word1.grapheme[:word1_grapheme_nonoverlap_end_idx+1] + word2.grapheme
				# arpabet_portmanteau1 = word1.arpabet_phoneme + word2.arpabet_phoneme[word2_arpabet_nonoverlap_start_idx:]
				# arpabet_portmanteau2 = word1.arpabet_phoneme[:word1_arpabet_nonoverlap_end_idx+1] + word2.arpabet_phoneme

				# If first word can be more easily reconstructed than the second, flip the ordering of the graphemes
				if word1_prob_given_dangling_graphs > word2_prob_given_dangling_graphs:
					grapheme_portmanteau1, grapheme_portmanteau2 = grapheme_portmanteau2, grapheme_portmanteau1
					arpabet_portmanteau1, arpabet_portmanteau2 = arpabet_portmanteau2, arpabet_portmanteau1

				# probability of a phoneme occurring with a particular occompanying grapheme is inversely proportional to the pun's quality; same is true for rhymes
				# basically, this is a roundabout way of identifying/discarding common prefixes/suffixes (commonly occurring graph/phone combinations at the start/end of words)
				word1_overlap_grapheme_phoneme_prob = cls.get_grapheme_phoneme_prob(word1.grapheme[word1_grapheme_overlap_start_idx:word1_grapheme_overlap_end_idx+1], word1_arpabet_overlap, pronunciation_dictionary)
				word2_overlap_grapheme_phoneme_prob = cls.get_grapheme_phoneme_prob(word2.grapheme[word2_grapheme_overlap_start_idx:word2_grapheme_overlap_end_idx+1], word2_arpabet_overlap, pronunciation_dictionary)
				overlap_grapheme_phoneme_prob = max(word1_overlap_grapheme_phoneme_prob, word2_overlap_grapheme_phoneme_prob)

				# pdb.set_trace()

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
					overlap_grapheme_phoneme_prob
					)
				return portmanteau, 0, 'portmanteau found!'

		# failed to find any overlaps meeting the 'max_overlap_dist' criteria, so return with default error message
		return portmanteau, status, message

	@staticmethod
	def get_prob_word_given_subgrapheme(subgrapheme, side, pronunciation_dictionary):
		'''
		Given that we know a grapheme contains a certain substring at a certain position, what is the probability that we can guess the grapheme?

		Need to have access to the PronounciationDictionary... how to do it?

		This _may_ be prohibitively innefficient to run every time... not sure

		Should be called with *dangling* grapheme substring

		As always, indices are inclusive, so add 1 to end_idx

		side = 'head' or 'tail'
		'''
		if side == 'head':
			subgraph_matches = [1 if subgrapheme == grapheme[:len(subgrapheme)] else 0 for grapheme in pronunciation_dictionary.grapheme_to_word_dict.iterkeys()]
		elif side == 'tail':
			subgraph_matches = [1 if subgrapheme == grapheme[-len(subgrapheme):] else 0 for grapheme in pronunciation_dictionary.grapheme_to_word_dict.iterkeys()]
		else:
			raise "Argument 'side' must be either 'head' or 'tail'"
		return 1.0 / sum(subgraph_matches)

	# @staticmethod
	# def get_subphoneme_prob(subphoneme, pronunciation_dictionary):
	# 	'''
	# 	How common is it for this particular phonetic element to occur at the start of end of a word?
	# 	If it's extremely common, then it's probably a (garbage) common prefix/suffix
	# 	'''
	# 	subphoneme_matches = [1 if subphoneme == word.arpabet_phoneme[:len(subphoneme)] or subphoneme == word.arpabet_phoneme[:len(subphoneme)] else 0 for word in pronunciation_dictionary.grapheme_to_word_dict.itervalues()]
	# 	return 1.0 / sum(subphoneme_matches)

	@staticmethod
	def get_grapheme_phoneme_prob(subgrapheme, subphoneme, pronunciation_dictionary):
		'''
		How common is it for this particular graphic+phonetic element to occur at the start of end of a word?
		If it's extremely common, then it's probably a (garbage) common prefix/suffix
		'''
		subgrapheme_matches_head = np.array([1 if subgrapheme == grapheme[:len(subgrapheme)] else 0 for grapheme in pronunciation_dictionary.grapheme_to_word_dict.iterkeys()])
		subgrapheme_matches_tail = np.array([1 if subgrapheme == grapheme[-len(subgrapheme):] else 0 for grapheme in pronunciation_dictionary.grapheme_to_word_dict.iterkeys()])
		subphoneme_matches_head = np.array([1 if subphoneme == word.arpabet_phoneme[:len(subphoneme)] else 0 for word in pronunciation_dictionary.grapheme_to_word_dict.itervalues()])
		subphoneme_matches_tail = np.array([1 if subphoneme == word.arpabet_phoneme[-len(subphoneme):] else 0 for word in pronunciation_dictionary.grapheme_to_word_dict.itervalues()])
		# if subgrapheme in ('un','gry','gri','grie'):
		# if subgrapheme in ('un','gry','gri','grie'):
			# pdb.set_trace()
		return 1.0 * ((subgrapheme_matches_head & subphoneme_matches_head) | (subgrapheme_matches_tail & subphoneme_matches_tail)).sum() / len(pronunciation_dictionary.grapheme_to_word_dict)

	def __repr__(self):
		return '''
		-------------------------------------------------------------------------------
		# Word Combination: {} + {}
		# Grapheme Portmanteau: {} ({})
		# Phoneme Portmanteau: {} ({})
		# Phoneme Distance: {}
		# Grapheme+Phoneme Probability: {}
		# Overlapping Phones: {}
		# Overlapping Vowel Phones: {}
		# Overlapping Consonant Phones: {}
		-------------------------------------------------------------------------------
		'''.format(self.word1.grapheme,
			self.word2.grapheme,
			self.grapheme_portmanteau1,
			self.grapheme_portmanteau2,
			'-'.join(self.arpabet_portmanteau1), # represented internally as a list, so collapse the list to a string
			'-'.join(self.arpabet_portmanteau2), # represented internally as a list, so collapse the list to a string
			self.overlap_distance,
			round(self.overlap_grapheme_phoneme_prob, 5),
			self.n_overlapping_phones,
			self.n_overlapping_vowel_phones,
			self.n_overlapping_consonant_phones)

	def __str__(self):
		return '{} ({}/{})'.format(self.grapheme_portmanteau1, self.word1.grapheme, self.word2.grapheme)

	def ordering_criterion(self):
		'''Smaller values are "better" portmanteaus'''
		return (-self.n_overlapping_phones, self.overlap_distance, self.overlap_grapheme_phoneme_prob)
		# return (-self.n_overlapping_phones, self.overlap_grapheme_phoneme_prob)
		# return (self.overlap_grapheme_phoneme_prob, -self.n_overlapping_phones, -self.n_overlapping_vowel_phones, self.overlap_distance)