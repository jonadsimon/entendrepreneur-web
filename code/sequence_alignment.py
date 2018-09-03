import numpy as np

class SequenceAlignment(object):
	'''
	Class for storing general m2m alignments, and allowing for backwards index-based mapping

	Possible for graph to map to null-phone
	NOT possible for graph to map to null-phone

	Possible for multiple graphs to map to 1 phone
	Possible for multiple phones to map to 1 graph

	Pretty sure that I can cover both directions...

	Used for
	  (1) grapheme to arpabet phhoneme alignments
	  (2) arpabet phoneme to vectorizable phoneme alignments

	Inputs: list of 2-tuples of n-tuples for variable n

	[    |  |   |    |    ] --> [    |  |   |    |    ]
	[..., ..., ... ,...] --> [..., ..., ... ,...]


	[(g1),(g2,g3),(g5)] --> [(p1,p2),(p3),(_)]
	'''
	
	def __init__(self, seq1, seq2):
		'''
		Initize using a dictionary mapping graphemes to phonemes

		seq1 - a length-n list of tuples
		seq2 - a length-n list of tuples
		'''

		if len(seq1) != len(seq2):
			raise Exception('Sequences must be the same length: {} != {}'.format(len(seq1),len(seq2)))

		self.seq1 = seq1
		self.seq2 = seq2
		
	def subseq2_to_subseq1(self, start_idx, end_idx):
		'''
		Given a start index and end index (both inclusive) in seq2 find the corresponding indices in the first sequence

		The alignement is performed in chunks, it's possible that the indices fall in the MIDDLE of a chunk. In this case, return an error
		'''

		# Verify that neither index falls in the middle of a chunk

		chunk_lengths = map(len, self.seq2)
		valid_end_inds = np.cumsum(chunk_lengths) - 1
		valid_start_inds = np.cumsum(chunk_lengths) - chunk_lengths

		if start_idx not in valid_start_inds:
			raise Exception('\'start_idx\' falls in the middle of a sequence chunk')

		if end_idx not in valid_end_inds:
			raise Exception('\'end_idx\' falls in the middle of a sequence chunk')

		# Alignment is super easy this way, can't believe I didn't think of it last time...
		# Include null-graphs at the boundaries
		start_chunk_idx = np.where(valid_start_inds == start_idx)[0].min()
		end_chunk_idx = np.where(valid_end_inds == end_idx)[0].max()

		subseq1 = sum(map(list, self.seq1[start_chunk_idx:end_chunk_idx+1]), [])
	
		return subseq1

	def subseq2_inds_to_subseq1_inds(self, start_idx, end_idx):
		'''
		Given a start index and end index (both inclusive) in seq2 find the corresponding indices in the first sequence
		'''

		# Verify that neither index falls in the middle of a chunk

		chunk_lengths = map(len, self.seq2)
		valid_end_inds = np.cumsum(chunk_lengths) - 1
		valid_start_inds = np.cumsum(chunk_lengths) - chunk_lengths

		if start_idx not in valid_start_inds:
			raise Exception('\'start_idx\' falls in the middle of a sequence chunk')

		if end_idx not in valid_end_inds:
			raise Exception('\'end_idx\' falls in the middle of a sequence chunk')

		# Alignment is super easy this way, can't believe I didn't think of it last time...
		# Include null-graphs at the boundaries
		start_chunk_idx = np.where(valid_start_inds == start_idx)[0].min()
		end_chunk_idx = np.where(valid_end_inds == end_idx)[0].max()

		# How many seq1 characters precede start_chunk_idx?
		# How many seq1 characters precede end_chunk_idx?
		subseq1_start_idx = sum(map(len, self.seq1[:start_chunk_idx]))
		subseq2_end_idx = sum(map(len, self.seq1[:end_chunk_idx+1])) - 1

		return subseq1_start_idx, subseq2_end_idx