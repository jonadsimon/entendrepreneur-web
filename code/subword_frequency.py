import dill # required to pickle lambda functions

class SubwordFrequency(object):
	'''
	Class which extends the CMU Pronunciation dictionary by mapping graphemes to Word objects
	'''
	def __init__(self, 
			        subgrapheme_head_counts, 
			        subgrapheme_tail_counts, 
			        subgrapheme_counts, 
			        subphoneme_head_counts, 
			        subphoneme_tail_counts, 
			        subphoneme_counts, 
			        subword_head_counts, 
			        subword_tail_counts, 
			        subword_counts,
			        vocab_size):
		'''
		Initialize using a dictionary mapping graphemes to phonemes
		'''
		self.subgrapheme_head_counter = subgrapheme_head_counts
		self.subgrapheme_tail_counter = subgrapheme_tail_counts
		self.subgrapheme_counter = subgrapheme_counts

		self.subphoneme_head_counter = subphoneme_head_counts
		self.subphoneme_tail_counter = subphoneme_tail_counts
		self.subphoneme_counter = subphoneme_counts

		self.subword_head_counter = subword_head_counts
		self.subword_tail_counter = subword_tail_counts
		self.subword_counter = subword_counts

		self.vocab_size = vocab_size

	def get_subgrapheme_frequency(self, grapheme, side='all'):
		if side == 'head':
			return self.subgrapheme_head_counter[grapheme]
		elif side == 'tail':
			return self.subgrapheme_tail_counter[grapheme]
		elif side == 'all':
			return self.subgrapheme_counter[grapheme]
		else:
			raise "Argument 'side' must be either 'head', 'tail', or 'all'"

	def get_subphoneme_frequency(self, phoneme, side='all'):
		if side == 'head':
			return self.subphoneme_head_counter[phoneme]
		elif side == 'tail':
			return self.subphoneme_tail_counter[phoneme]
		elif side == 'all':
			return self.subphoneme_counter[phoneme]
		else:
			raise "Argument 'side' must be either 'head', 'tail', or 'all'"

	def get_subword_frequency(self, grapheme, phoneme, side='all'):
		if side == 'head':
			return self.subword_head_counter[(grapheme,phoneme)]
		elif side == 'tail':
			return self.subword_tail_counter[(grapheme,phoneme)]
		elif side == 'all':
			return self.subword_counter[(grapheme,phoneme)]
		else:
			raise "Argument 'side' must be either 'head', 'tail', or 'all'"

	def save(self, filename):
		'''
		Save the object to a file (just pickle it)
		'''
		with open(filename, 'wb') as outfile:
			dill.dump(self, outfile)

	@staticmethod
	def load(filename):
		'''
		Load a saved object from a file
		'''
		with open(filename, 'rb') as infile:
			return dill.load(infile)