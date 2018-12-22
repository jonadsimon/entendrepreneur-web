from nltk.corpus import cmudict
from collections import defaultdict
import sys
sys.path.insert(0, '../code')
from word import Word
from pronunciation_dictionary import PronunciationDictionary
from sequence_alignment import SequenceAlignment
from subword_frequency import SubwordFrequency

# PronunciationDictionary produced by convert_m2m_aligner_results_to_pronunciation_dictionary.py
pd = PronunciationDictionary.load('../data/pronunciation_dictionary.pkl')

def graph_chunks_to_key(g):
    return ''.join(sum(map(list, g), []))

def phone_chunks_to_key(p):
    return tuple(sum(map(list, p), []))

# Counters for storing the frequencies of each subgrapheme/subphoneme/subword
# Smooth by using default value of 1 to avoid probability singularities when string is not present

subgrapheme_head_counts = defaultdict(lambda: 1)
subgrapheme_tail_counts = defaultdict(lambda: 1)
subgrapheme_counts = defaultdict(lambda: 1)

subphoneme_head_counts = defaultdict(lambda: 1)
subphoneme_tail_counts = defaultdict(lambda: 1)
subphoneme_counts = defaultdict(lambda: 1)

subword_head_counts = defaultdict(lambda: 1)
subword_tail_counts = defaultdict(lambda: 1)
subword_counts = defaultdict(lambda: 1)

# Total number of words in the vocabulary
vocab_size = 0

for grapheme,word in pd.grapheme_to_word_dict.iteritems():
    graph_chunks = word.grapheme_to_phoneme_alignment.seq1
    phone_chunks = word.grapheme_to_phoneme_alignment.seq2
    vocab_size += 1
    # Only store substrings comprised of at most 5 aligned chunks
    # Anything longer than that is rare enough that the default count of 1 is a good approximation
    for k in range(1,6):
        for i in range(len(graph_chunks)-k+1):
            g = graph_chunks_to_key(graph_chunks[i:i+k])
            p = phone_chunks_to_key(phone_chunks[i:i+k])
            subgrapheme_counts[g] += 1
            subphoneme_counts[p] += 1
            subword_counts[(g,p)] += 1
            if i == 0: # head
                subgrapheme_head_counts[g] += 1
                subphoneme_head_counts[p] += 1
                subword_head_counts[(g,p)] += 1
            if i + k == len(graph_chunks): # tail
                subgrapheme_tail_counts[g] += 1
                subphoneme_tail_counts[p] += 1
                subword_tail_counts[(g,p)] += 1

# Save results as SubwordFrequency object
sf = SubwordFrequency(
        subgrapheme_head_counts,
        subgrapheme_tail_counts,
        subgrapheme_counts,
        subphoneme_head_counts,
        subphoneme_tail_counts,
        subphoneme_counts,
        subword_head_counts,
        subword_tail_counts,
        subword_counts,
        vocab_size)
sf.save('../data/subword_frequency.pkl')
