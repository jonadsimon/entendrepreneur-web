import cPickle as pkl

class PronunciationDictionary(object):
	'''
	---------------
	# DESCRIPTION #
	---------------
	The PronunciationDictionary extends the CMU Pronuncing Dictionary provided by the nltk library
	by mapping graphemes to Word objects containing the grapheme/phoneme alignment as well as the phoneme itself

	-------------------
	# CLASS VARIABLES #
	-------------------
	grapheme_to_word_dict, Dict[String => Word] : dictionary mapping grapheme strings to their associated Word objects

	-----------------
	# CLASS METHODS #
	-----------------
	get_word : returns the Word associated with the input grapheme
	save : pickles and saves the current PronunciationDictionary object
	load : loads a pickled PronunciationDictionary object
	'''
	def __init__(self, word_list):
		self.grapheme_to_word_dict = dict((word.grapheme, word) for word in word_list)

	def get_word(self, grapheme):
		'''
		Return the Word object associated with the input grapheme
		If no associated Word exists, then return None
		'''
		return self.grapheme_to_word_dict.get(grapheme, None)

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
