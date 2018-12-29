import gensim
import sys
sys.path.insert(0, '../code')
from global_constants import REPO_HOME, MAX_VOCAB
from fasttext_vector_element_table import FasttextVectorElement

word_id = Column(Integer, ForeignKey('words.id'), )
index = Column(Integer)
value = Column(Float) # these values are NOT L2-normalized, need to do that on each call to most_similar

fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(REPO_HOME+'data/word_vectors/wiki-news-300d-1M.vec', limit=MAX_VOCAB)

vector_element_list = []
for grapheme_idx, grapheme in enumerate(fasttext_model.iterkeys()):
    # Find the entry in the Word table corrersponing to this grapheme
    # If it doesn't exist, then leave the foreign_key field null
    #
    # Wait... how to deal with casing discrepancies? Multiple vectors with the same key?
    # Maybe should remove this dependence on word... probably the best way to handle things
    for vector_element_idx, vector_element in enumerate(fasttext_model.vectors[grapheme_idx]):
        new_vector_element = FasttextVectorElement()



def populate_fasttext_vector_element_table(session):

    # Counters for storing the frequencies of each subgrapheme
    # Smooth by using default value of 1 to avoid probability singularities when string is not present
    subgrapheme_head_counts = defaultdict(lambda: 1)
    subgrapheme_tail_counts = defaultdict(lambda: 1)
    subgrapheme_counts = defaultdict(lambda: 1)

    for grapheme in cmu_dict.iterkeys():
        # Only store subgraphemes up to a length of 5
        # Anything longer than that is rare enough that the default count of 1 is a good approximation
        for k in range(1,6):
            for i in range(len(grapheme)-k+1):
                g = grapheme[i:i+k]
                subgrapheme_counts[g] += 1
                if i == 0: # head
                    subgrapheme_head_counts[g] += 1
                if i + k == len(grapheme): # tail
                    subgrapheme_tail_counts[g] += 1

    # Now iterate through the subgraphemes, and add them to the SubgraphemeFrequency table
    subgrapheme_frequency_list = []
    for g in subgrapheme_counts.iterkeys():
        new_subgrapheme_frequency = SubgraphemeFrequency(grapheme=g,
                                                        frequency=subgrapheme_counts[g],
                                                        frequency_head=subgrapheme_head_counts[g],
                                                        frequency_tail=subgrapheme_tail_counts[g])
        subgrapheme_frequency_list.append(new_subgrapheme_frequency)

    # Add the generated SubgraphemeFrequency objects to the SubgraphemeFrequency table, and commit the changes
    session.add_all(subgrapheme_frequency_list)
    session.commit()
