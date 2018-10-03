import cPickle as pkl


class PronunciationDictionary(object):
	'''
	Class which extends the CMU Pronunciation dictionary by mapping graphemes to Word objects
	'''
	def __init__(self, word_list):
		'''
		Initialize using a dictionary mapping graphemes to phonemes
		'''
		self.grapheme_to_word_dict = dict((word.grapheme, word) for word in word_list)

	def get_word(self, grapheme):
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