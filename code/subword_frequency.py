import dill # required to pickle lambda functions

class SubwordFrequency(object):
	'''
	---------------
	# DESCRIPTION #
	---------------
	The SubwordFrequency class stores information about the frequency of partial graphemes and
	phonemes appearing in the CMU Pronouncing Dictionary.

	-------------------
	# CLASS VARIABLES #
	-------------------
	subgrapheme_head_counter, Dict[String => Int] : frequency with which each subgrapheme occurred at the start of a grapheme
	subgrapheme_tail_counter, Dict[String => Int] : frequency with which each subgrapheme occurred at the end of a grapheme
	subgrapheme_counter, Dict[String => Int] : frequency with which each subgrapheme occurred within a grapheme

	subphoneme_head_counter, Dict[Array[String] => Int] : frequency with which each subphoneme occurred at the start of a phoneme
	subphoneme_tail_counter, Dict[Array[String] => Int] : frequency with which each subphoneme occurred at the end of a phoneme
	subphoneme_counter, Dict[Array[String] => Int] : frequency with which each subphoneme occurred within a phoneme

	subword_head_counter, Dict[(String,Array[String]) => Int] : frequency with which each subgrapheme/subphoneme pair occurred at the start of a word
	subword_tail_counter, Dict[(String,Array[String]) => Int] : frequency with which each subgrapheme/subphoneme pair occurred at the end of a word
	subword_counter, Dict[(String,Array[String]) => Int] : frequency with which each subgrapheme/subphoneme pair occurred within a word

	vocab_size, Int : number of graphemes in the CMU Pronouncing Dictionary

	-----------------
	# CLASS METHODS #
	-----------------
	get_subgrapheme_frequency : returns the frequency with which a given subgrapheme occurs in the CMU Pronouncing Dictionary
	get_subphoneme_frequency : returns the frequency with which a given subphoneme occurs in the CMU Pronouncing Dictionary
	get_subword_frequency : returns the frequency with which a given subgrapheme/subphoneme pair occurs in the CMU Pronouncing Dictionary
	save : pickles and saves the current SubwordFrequency object
	load : loads a pickled SubwordFrequency object
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
		'''
		---------------
		# DESCRIPTION #
		---------------
		Returns the frequencies with which particular subgraphemes occur in the CMU Pronouncing Dictionary.
		If the subgrapheme does not appear in the corpus, then return a frequency of 1

		----------
		# INPUTS #
		----------
		grapheme, String : subgrapheme to obtain the frequency of
		side, String : 'tail' (end-of-grapheme freqs), 'head' (start-of-grapheme freqs), or 'all' (within-grapheme freqs)
		'''
		if side == 'head':
			return self.subgrapheme_head_counter[grapheme]
		elif side == 'tail':
			return self.subgrapheme_tail_counter[grapheme]
		elif side == 'all':
			return self.subgrapheme_counter[grapheme]
		else:
			raise "Argument 'side' must be either 'head', 'tail', or 'all'"

	def get_subphoneme_frequency(self, phoneme, side='all'):
		'''
		---------------
		# DESCRIPTION #
		---------------
		Returns the frequencies with which particular subphonemes occur in the CMU Pronouncing Dictionary.
		If the subphoneme does not appear in the corpus, then return a frequency of 1

		----------
		# INPUTS #
		----------
		phoneme, Array[String] : subphoneme to obtain the frequency of
		side, String : 'tail' (end-of-phoneme freqs), 'head' (start-of-phoneme freqs), or 'all' (within-phoneme freqs)
		'''
		if side == 'head':
			return self.subphoneme_head_counter[phoneme]
		elif side == 'tail':
			return self.subphoneme_tail_counter[phoneme]
		elif side == 'all':
			return self.subphoneme_counter[phoneme]
		else:
			raise "Argument 'side' must be either 'head', 'tail', or 'all'"

	def get_subword_frequency(self, grapheme, phoneme, side='all'):
		'''
		---------------
		# DESCRIPTION #
		---------------
		Returns the frequencies with which particular subgrapheme/subphonemes pairs occur in the CMU Pronouncing Dictionary.
		If the subgrapheme/subphoneme pair does not appear in the corpus, then return a frequency of 1

		----------
		# INPUTS #
		----------
		grapheme, String : subgrapheme part of the subgrapheme/subphoneme pair to obtain the frequency of
		phoneme, Array[String] : subphoneme part of the subgrapheme/subphoneme pair to obtain the frequency of
		side, String : 'tail' (end-of-word freqs), 'head' (start-of-word freqs), or 'all' (within-word freqs)
		'''
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
