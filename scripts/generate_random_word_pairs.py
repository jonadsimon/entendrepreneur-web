from nltk.corpus import wordnet as wn
from itertools import product
from collections import Counter
import random

random.seed(0)

def get_most_common_pos(w):
	return Counter([l.pos() for l in wn.synsets(w)]).most_common(1)[0][0]

# chop off trailing 's' where possible
def get_stemmed_word(w):
	if w[-1] == 's' and wn.synsets(w[:-1]):
	  return w[:-1]
	else:
	 	return w

data_path = 'data/count_1w.txt'
with open(data_path) as infile:
	words = [line.strip().split()[0] for line in infile.readlines()]

proper_nouns_path = 'data/names_locations/common_proper_nouns.txt'
with open(proper_nouns_path) as infile:
	proper_nouns_set = set([line.strip() for line in infile.readlines()])

# only want the most common 10k words to keep the weird ones out
max_words = 10000
words = words[:max_words]

max_pairs = 1000
n_pairs = 0
pos_pairs = set([('a','n'),('s','n')])
with open('data/random_word_pairs.txt','w') as outfile:
	while n_pairs < max_pairs:
		w1, w2 = random.sample(words, 2)
		# To filter out junk, make sure that:
		# 1) both words are present in WordNet
		# 2) both words are at least four letters long
		# 3) words are not proper nounts (names/places)
		# 4) words are (adj, noun) pairs
		# 5) discard plurals
		if wn.synsets(w1) and wn.synsets(w2) and len(w1) > 3 and len(w2) > 3 and w1 not in proper_nouns_set and w2 not in proper_nouns_set:
			w1 = get_stemmed_word(w1)
			w2 = get_stemmed_word(w2)
			pos1 = get_most_common_pos(w1)
			pos2 = get_most_common_pos(w2)
			if (pos1,pos2) in pos_pairs:
				outfile.write('{} {}\n'.format(w1,w2))
				n_pairs += 1