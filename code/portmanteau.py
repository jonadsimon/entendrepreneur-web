class Portmanteau(Pun):
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
							overlap_distance):
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

	@classmethod
	def get_portmanteau(cls, word1, word2, pronunciation_dictionary):
		'''
		Attempts to create a Portmanteau out of the two words
		If successful, returns the Portmanteau
		If unnsuccessful, returns nil along with a descriptive error message

		We need the 'pronunciation_dictionary' in order to determine the probability of each grapheme
		'''

		# these are the default return values if no good overlaps are found
		portmanteau, status, message = None, 1, 'no ≤max_overlap_dist overlaps were found'

		min_word_len = min(len(word1.vectorizable_phoneme), len(word2.vectorizable_phoneme))
		for overlap_len in range(min_word_len):
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
					pass
				else:
					num_overlap_vowel_phones1 = sum([1 if phone in ARPABET_VOWELS else 0 for phone in word1_arpabet_overlap])
					num_overlap_consonant_phones1 = sum([1 if phone in ARPABET_CONSONANTS else 0 for phone in word1_arpabet_overlap])
					num_non_overlap_phones1 = len(word1_arpabet_nonoverlap)
					
					if num_overlap_vowel_phones < cls.min_overlap_vowel_phones:
						portmanteau, status, message = None, 1, 'word1 overlap does not have enough vowels'
						pass
					elif num_overlap_consonant_phones < cls.min_overlap_consonant_phones:
						portmanteau, status, message = None, 1, 'word1 overlap does not have enough consonants'
						pass
					elif num_non_overlap_phones < cls.min_non_overlap_phones:
						portmanteau, status, message = None, 1, 'word1 non-overlap does not have enough characters'
						pass

				try:
					word2_arpabet_overlap = word2.arpabet_phoneme_to_vectorizable_phoneme_alignment.subseq2_to_subseq1(0, word2_idx-1)
					word2_arpabet_nonoverlap = word2.arpabet_phoneme_to_vectorizable_phoneme_alignment.subseq2_to_subseq1(word2_idx, len(word2.vectorizable_phoneme)-1)
				except:
					portmanteau, status, message = None, 1, 'word2 vectorizable_phoneme could not be aligned with arpabet_phoneme'
					pass
				else:
					num_overlap_vowel_phones2 = sum([1 if phone in ARPABET_VOWELS else 0 for phone in word2_arpabet_overlap])
					num_overlap_consonant_phones2 = sum([1 if phone in ARPABET_CONSONANTS else 0 for phone in word2_arpabet_overlap])
					num_non_overlap_phones2 = len(word2_arpabet_nonoverlap)

					if num_overlap_vowel_phones2 < cls.min_overlap_vowel_phones:
						portmanteau, status, message = None, 1, 'word2 overlap does not have enough vowels'
						pass
					elif num_overlap_consonant_phones2 < cls.min_overlap_consonant_phones:
						portmanteau, status, message = None, 1, 'word2 overlap does not have enough consonants'
						pass
					elif num_non_overlap_phones2 < cls.min_non_overlap_phones:
						portmanteau, status, message = None, 1, 'word2 non-overlap does not have enough characters'
						pass

				# vectorizable_phoneme-to-arpabet_phoneme alignments worked *and* all arpabet constraints are met
				# possible for one grapheme alignment to work, but not the other, HOWEVER enforce that they BOTH work for ease of future logic
				# (this will throw away some otherwise fine portmanteaus, so return to handle this edge case later)
				try:
					word1_arpabet_overlap_start_idx, word1_arpabet_overlap_end_idx = word1.arpabet_phoneme_to_vectorizable_phoneme_alignment.subseq2_inds_to_subseq1_inds(word1_idx, len(word1.vectorizable_phoneme)-1)
					word1_grapheme_overlap_start_idx, word1_grapheme_overlap_end_idx = word1.grapheme_to_arpabet_phoneme_alignment.subseq2_to_subseq1(word1_arpabet_overlap_start_idx, word1_arpabet_overlap_end_idx)
				except:
					portmanteau, status, message = None, 1, 'word1 arpabet_phoneme could not be aligned with grapheme'
					pass
				
				try:
					word2_arpabet_overlap_start_idx, word2_arpabet_overlap_end_idx = word2.arpabet_phoneme_to_vectorizable_phoneme_alignment.subseq2_inds_to_subseq1_inds(0, word2_idx-1)
					word2_grapheme_overlap_start_idx, word2_grapheme_overlap_end_idx = word2.grapheme_to_arpabet_phoneme_alignment.subseq2_to_subseq1(word2_arpabet_overlap_start_idx, word2_arpabet_overlap_end_idx)
				except:
					portmanteau, status, message = None, 1, 'word2 arpabet_phoneme could not be aligned with grapheme'
					pass

				# All alignments and min-char requirements have been met, so create the Portmanteau, and return

				# pick the graphemetric representation such that the constituent words are most easily reconstruble
				# i.e. use the innards of the word which is most easily predicted from its dangling phones

				# may be rife with OBO errors, do another round to double-check afterwards
				word1_grapheme_nonoverlap_start_idx, word1_grapheme_nonoverlap_end_idx = word1.grapheme_to_arpabet_phoneme_alignment.subseq2_inds_to_subseq1_inds(0, word1_arpabet_overlap_start_idx-1)
				word2_grapheme_nonoverlap_start_idx, word2_grapheme_nonoverlap_end_idx = word2.grapheme_to_arpabet_phoneme_alignment.subseq2_inds_to_subseq1_inds(word2_arpabet_overlap_end_idx, len(word2.arpabet_phoneme)-1)

				word1_prob_given_dangling_graphs = self.get_word_prob_from_subgraph(word1.grapheme, word1_grapheme_nonoverlap_start_idx, word1_grapheme_nonoverlap_end_idx, pronunciation_dictionary)
				word2_prob_given_dangling_graphs = self.get_word_prob_from_subgraph(word2.grapheme, word2_grapheme_nonoverlap_start_idx, word2_grapheme_nonoverlap_end_idx, pronunciation_dictionary)

				grapheme_portmanteau1 = word1.grapheme + word2.grapheme[word2_grapheme_nonoverlap_start_idx:]
				grapheme_portmanteau2 = word1.grapheme[:word2_grapheme_nonoverlap_end_idx] + word2.grapheme
				arpabet_portmanteau1 = word1.arpabet_phoneme + word2.arpabet_phoneme[word2_arpabet_nonoverlap_start_idx:]
				arpabet_portmanteau2 = word1.arpabet_phoneme[:word2_arpabet_nonoverlap_end_idx] + word2.arpabet_phoneme

				# If first word can be more easily reconstructed than the second, flip the ordering of the graphemes
				if word1_prob_given_dangling_graphs > word2_prob_given_dangling_graphs:
					grapheme_portmanteau1, grapheme_portmanteau2 = grapheme_portmanteau2, grapheme_portmanteau1
					arpabet_portmanteau1, arpabet_portmanteau2 = arpabet_portmanteau2, arpabet_portmanteau1

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
					overlap_distance
					)
				return portmanteau, 0, 'portmanteau found!'

		# failed to find any overlaps meeting the 'max_overlap_dist' criteria, so return with default error message
		return portmanteau, status, message

		def get_word_prob_from_subgraph(self, grapheme, start_idx, end_idx, pronunciation_dictionary):
			'''
			Given that we know a grapheme contains a certain substring at a certain position, what is the probability that we can guess the grapheme?

			Need to have access to the PronounciationDictionary... how to do it?

			This _may_ be prohibitively innefficient to run every time... not sure

			Should be called with *dangling* grapheme substring
			'''
			subgraph = grapheme[start_idx:end_idx]
			subgraph_matches = [1 if subgraph == grapheme[start_idx:end_idx] else 0 for grapheme in pronunciation_dictionary.grapheme_to_word_dict.iterkeys()]
			return 1.0*sum(subgraph_matches) / len(subgraph_matches)

		def __repr__(self):
			return '''
			---------------------------------------------------------------
			Word Combination: {} + {}
			Grapheme Portmanteau: {} ({})
			Phoneme Portmanteau: {} ({})
			Phoneme Distance: {}
			# Overlapping Phones: {}
			# Overlapping Vowel Phones: {}
			# Overlapping Consonant Phones: {}
			---------------------------------------------------------------
			'''.format(self.word1.grapheme,
				self.word2.grapheme,
				self.grapheme_portmanteau1,
				self.grapheme_portmanteau2,
				self.phoneme_portmanteau1,
				self.phoneme_portmanteau2,
				self.overlap_distance,
				self.n_overlapping_phones,
				self.n_overlapping_vowel_phones,
				self.n_overlapping_consonant_phones)

		def __str__(self):
			return '{} ({}/{})'.format(self.grapheme_portmanteau1, self.word1.grapheme, self.word2.grapheme)

    def ordering_criterion(self):
      '''Smaller values are "better" portmanteaus'''
      return (-self.n_overlapping_phones, -self.n_overlapping_vowel_phones, self.phoneme_distance)