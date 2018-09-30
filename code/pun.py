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
	def get_pun(cls, word1, word2, pronunciation_dictionary):
		pass

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
			# need to explicitly add 'len(grapheme)' to the negative-indexing to handle 0-cases
			subgraph_matches = [1 if subgrapheme == grapheme[len(grapheme)-len(subgrapheme):] else 0 for grapheme in pronunciation_dictionary.grapheme_to_word_dict.iterkeys()]
		else:
			raise "Argument 'side' must be either 'head' or 'tail'"
		
		return 1.0 / sum(subgraph_matches)


	@staticmethod
	def get_prob_word_given_subphoneme_and_grapheme_length(grapheme, subphoneme, pronunciation_dictionary):
		'''
		For all words whose graphemes are as-short-or-shorter than this word, how many of them end in this phoneme?
		Need to condition on word length so that short words whose phonemes comprise a large % of the word (even if they're common) aren't penalized

		# Need to think more about this one...
		'''
		subphoneme_matches = [1 if subphoneme == word.arpabet_phoneme[-len(subphoneme):] and len(word.grapheme) <= len(grapheme) else 0 for word in pronunciation_dictionary.grapheme_to_word_dict.itervalues()]
		return 1.0 / sum(subphoneme_matches)

	@staticmethod
	def get_grapheme_phoneme_prob(subgrapheme, subphoneme, pronunciation_dictionary):
		'''
		How common is it for this particular graphic+phonetic element to occur at the start of end of a word?
		If it's extremely common, then it's probably a (garbage) common prefix/suffix
		'''
		subgrapheme_matches_head = np.array([1 if subgrapheme == grapheme[:len(subgrapheme)] else 0 for grapheme in pronunciation_dictionary.grapheme_to_word_dict.iterkeys()])
		# need to explicitly add 'len(grapheme)' to the negative-indexing to handle 0-cases
		subgrapheme_matches_tail = np.array([1 if subgrapheme == grapheme[len(grapheme)-len(subgrapheme):] else 0 for grapheme in pronunciation_dictionary.grapheme_to_word_dict.iterkeys()])
		subphoneme_matches_head = np.array([1 if subphoneme == word.arpabet_phoneme[:len(subphoneme)] else 0 for word in pronunciation_dictionary.grapheme_to_word_dict.itervalues()])
		# need to explicitly add 'len(word.arpabet_phoneme)' to the negative-indexing to handle 0-cases
		subphoneme_matches_tail = np.array([1 if subphoneme == word.arpabet_phoneme[len(word.arpabet_phoneme)-len(subphoneme):] else 0 for word in pronunciation_dictionary.grapheme_to_word_dict.itervalues()])
		return 1.0 * ((subgrapheme_matches_head & subphoneme_matches_head) | (subgrapheme_matches_tail & subphoneme_matches_tail)).sum() / len(pronunciation_dictionary.grapheme_to_word_dict)

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