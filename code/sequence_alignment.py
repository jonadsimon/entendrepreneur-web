import numpy as np

class SequenceAlignment(object):
	'''
	---------------
	# DESCRIPTION #
	---------------
	The SequenceAlignment class encodes the element-wise alignment between two sequences,
	allowing for easy mapping between constituent subsequences.

	For example for the alignment grapheme/phone pair for the word "cognac":
	seq1 = [('c',), ('o',), ('g',), ('n',), ('a',), ('c',)]
	seq2 = [(u'K',), (u'OW1',), (), (u'N', u'Y'), (u'AE2',), (u'K',)]

	Where:
	subseq2_to_subseq1(0, 1) returns : ['c', 'o', 'g']
	subseq2_to_subseq1(0, 2) returns : Exception: 'end_idx' falls in the middle of a sequence chunk
	subseq2_to_subseq1(0, 3) returns : ['c', 'o', 'g', 'n']
	...
	subseq2_to_subseq1(2, 3) returns : ['g', 'n']
	subseq2_to_subseq1(2, 4) returns : ['g', 'n', 'a']
	subseq2_to_subseq1(2, 5) returns : ['g', 'n', 'a', 'c']

	Note that contiguous silent letters are always included in the returned output
	This is why subseq2_to_subseq1(0, 1) returns ['c', 'o', 'g'] rather than ['c', 'o']
	and why subseq2_to_subseq1(2, 3) returns ['g', 'n'] rather than ['n']

	-------------------
	# CLASS VARIABLES #
	-------------------
	seq1, Array[Tuple[String]] : first sequence in the alignment, subdivided into aligned "chunks" (typically a grapheme)
	seq2, Array[Tuple[String]] : second sequence in the alignment, subdivided into aligned "chunks" (typically a phoneme)

	-----------------
	# CLASS METHODS #
	-----------------
	subseq2_to_subseq1 : given a start_idx and end_idx
	get_subphoneme_frequency : returns the frequency with which a given subphoneme occurs in the CMU Pronouncing Dictionary
	get_subword_frequency : returns the frequency with which a given subgrapheme/subphoneme pair occurs in the CMU Pronouncing Dictionary
	save : pickles and saves the current SubwordFrequency object
	load : loads a pickled SubwordFrequency object
	'''

	def __init__(self, seq1, seq2):
		if len(seq1) != len(seq2):
			raise Exception('Sequences must be the same length: {} != {}'.format(len(seq1),len(seq2)))

		self.seq1 = seq1
		self.seq2 = seq2

	def subseq2_to_subseq1(self, start_idx, end_idx):
		'''
		---------------
		# DESCRIPTION #
		---------------
		Given a subsequence of seq2 denoted by a pair of (inclusive) indices, returns the corresponding subsequence of seq1
		If either the start or end index falls in the middle of a paired-phoneme chunk, returns an error
		See class description for details

		----------
		# INPUTS #
		----------
		start_idx, Int : index denoted the (inclusive) start of the seq2 subsequence to be mapped
		end_idx, Int : index denoted the (inclusive) end of the seq2 subsequence to be mapped

		-----------
		# OUTPUTS #
		-----------
		subseq1, Array[String] : the corresponding subsequence of seq1
		'''

		# Verify that neither index falls in the middle of a chunk

		chunk_lengths = map(len, self.seq2)
		valid_end_inds = np.cumsum(chunk_lengths) - 1
		valid_start_inds = np.cumsum(chunk_lengths) - chunk_lengths

		if start_idx not in valid_start_inds:
			raise Exception('\'start_idx\' falls in the middle of a sequence chunk')

		if end_idx not in valid_end_inds:
			raise Exception('\'end_idx\' falls in the middle of a sequence chunk')

		# Include null-graphs at the boundaries
		start_chunk_idx = np.where(valid_start_inds == start_idx)[0].min()
		end_chunk_idx = np.where(valid_end_inds == end_idx)[0].max()

		subseq1 = sum(map(list, self.seq1[start_chunk_idx:end_chunk_idx+1]), [])

		return subseq1

	def subseq2_inds_to_subseq1_inds(self, start_idx, end_idx):
		'''
		---------------
		# DESCRIPTION #
		---------------
		Given a subsequence of seq2 denoted by a pair of (inclusive) indices, returns a pair of (inclusive) indices for seq1
		If either the start or end index falls in the middle of a paired-phoneme chunk, returns an error
		See class description for details

		----------
		# INPUTS #
		----------
		start_idx, Int : index denoted the (inclusive) start of the seq2 subsequence to be mapped
		end_idx, Int : index denoted the (inclusive) end of the seq2 subsequence to be mapped

		-----------
		# OUTPUTS #
		-----------
		subseq1_start_idx, Int : index denoted the (inclusive) start of the corresponding seq1 subsequence
		subseq2_start_idx, Int : index denoted the (inclusive) end of the corresponding seq2 subsequence
		'''

		# Verify that neither index falls in the middle of a chunk

		chunk_lengths = map(len, self.seq2)
		valid_end_inds = np.cumsum(chunk_lengths) - 1
		valid_start_inds = np.cumsum(chunk_lengths) - chunk_lengths

		if start_idx not in valid_start_inds:
			raise Exception('\'start_idx\' falls in the middle of a sequence chunk')

		if end_idx not in valid_end_inds:
			raise Exception('\'end_idx\' falls in the middle of a sequence chunk')

		# Include null-graphs at the boundaries
		start_chunk_idx = np.where(valid_start_inds == start_idx)[0].min()
		end_chunk_idx = np.where(valid_end_inds == end_idx)[0].max()

		# How many seq1 characters precede start_chunk_idx?
		# How many seq1 characters precede end_chunk_idx?
		subseq1_start_idx = sum(map(len, self.seq1[:start_chunk_idx]))
		subseq2_end_idx = sum(map(len, self.seq1[:end_chunk_idx+1])) - 1

		return subseq1_start_idx, subseq2_end_idx
