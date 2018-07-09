from gensim import matutils
import numpy as np
import cPickle as pkl

class WordVectorNearestNeighbors(object):
	'''
	Class which implements a small part of the functionality of gensim.models.KeyedVectors
	Necessary if we want to query a custom subset of word vectors for nearest-neighbor proximity

	Capable of saving and loading the contents for later use
	'''
	def __init__(self, word_to_vec_dict):
		'''
		Initize using a dictionary mapping word strings to (normalized) vectors
		'''
		self.words = np.array(word_to_vec_dict.keys())
		self.vectors_mat = np.array(word_to_vec_dict.values())

	def get_nearest_neighbors(self, word, num_neighbors=10):
		'''
		Find the 'num_neighbors' nearest neighbor vectors, and return them in order from closest to farthest
		This set of neighbors includes the word itself, which is definitionally its own nearest neighbor
		'''
		word_vector = self.vectors_mat[self.words.index(word)]
		cosine_sims = self.vectors_mat.dot(word_vector)
		neighbor_inds = matutils.argsort(cosine_sims, topn=num_neighbors, reverse=True)
		return self.words[neighbor_inds]

	def save(self, filename):
		'''
		Save the object to a file (just pickle it)
		'''
		with open(filename, 'wb') as outfile:
			pkl.dump(self, outfile)

	@staticmethod
	def load(filename):
		'''
		Load a saved object from a file
		'''
		with open(filename, 'rb') as infile:
			return pkl.load(infile)


class PronunciationDictionary(object):
	'''
	Class which extends the CMU Pronunciation dictionary to map to L2P alignments as well as phonemes
	Necessary if we want to query a custom subset of word vectors for nearest-neighbor proximity

	To cut down on ambiguity, we assume that there exists only a single phoneme (and alignment) per grapheme
	The corresponds to taking the first phoneme entry in the CMU Pronunciation dict
	'''
	def __init__(self, grapheme_to_phoneme_dict, grapheme_to_alignment_dict):
		'''
		Initize using a dictionary mapping graphemes to phonemes
		'''
		self.grapheme_to_phoneme_dict = grapheme_to_phoneme_dict
		self.grapheme_to_alignment_dict = grapheme_to_alignment_dict

	def get_phoneme(self, grapheme):
		return self.grapheme_to_phoneme_dict.get(grapheme, None)

	def get_alignment(self, grapheme):
		return self.grapheme_to_alignment_dict.get(grapheme, None)

	def save(self, filename):
		'''
		Save the object to a file (just pickle it)
		'''
		with open(filename, 'wb') as outfile:
			pkl.dump(self, outfile)

	@staticmethod
	def load(filename):
		'''
		Load a saved object from a file
		'''
		with open(filename, 'rb') as infile:
			return pkl.load(infile)


class GraphemePhonemeAlignment(object):
	'''
	L2P alignment object capable of mapping from subphonemes to corresponding to subgraphemes
	'''
	def __init__(self, m2m_aligner_output):
		'''
		Initize using a dictionary mapping graphemes to phonemes
		'''
		self.aligned_pairs = self.m2m_aligner_to_tuples(m2m_aligner_output)

	def m2m_aligner_to_tuples(self, m2m_aligner_output):
		'''
		Converts an alignment produced by the m2m-aligner to a list of aligned graph-phone pairs
		'''
		return zip(*map(lambda x: x.strip('|').split('|'), str(m2m_aligner_output).strip().split('\t')))
		
	def subphoneme_to_subgrapheme(self, phone_i1, phone_i2):
		'''
		Given a pair of (inclusive) phone indexes, find the grapheme that corresponds to that phoneme
		
		If the index lands in the middle of a double-phoneme, return an error message
		This can be used to forbid alignments which do not have corresponding unique grapheme alignments
		(for most of these, can get around it by cutting off the subphoneme one phone sooner)

		Strings are short enough, don't worry about premature breaking; it confuses the logic too much
		'''
		subgrapheme = ''
		subphoneme_idx = 0
		for graph,phone in self.aligned_pairs:

			# First phone lands in the middle of a middle of a double-phone
			if subphoneme_idx == phone_i1 - 1 and ':' in phone:
				raise Exception('1st index falls in the middle of double-phone')

			# Last phone lands in the middle of a double-phone
			if subphoneme_idx == phone_i2 and ':' in phone:
				raise Exception('2nd index falls in the middle of double-phone')

			# Add the graph if:
			# 1) subphoneme_idx is within the bounds of the phoneme
			# 2) subphoneme_idx is *just* outside the bounds of the phoneme *and* phone is null
			# 3) the phone is *not* a null phone at the start of the graph
			if (phone_i1 <= subphoneme_idx <= phone_i2 or (subphoneme_idx == phone_i2 + 1 and phone == '_')) and not (subgrapheme == '' and phone == '_'):
				# if it's a null graph, adding it does nothing
				# if it's a double-graph, drop the colon
				subgrapheme += graph.replace('_','').replace(':','')
			
			# is double-phone
			if ':' in phone:
				subphoneme_idx += 2
			# is not null-phone
			elif phone != '_':
				subphoneme_idx += 1
		
		return subgrapheme
