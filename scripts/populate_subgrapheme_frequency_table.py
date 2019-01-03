from nltk.corpus import cmudict
from collections import defaultdict
import sys
sys.path.insert(0, '../code')
from subgrapheme_frequency_table import SubgraphemeFrequency

# CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

def populate_subgrapheme_frequency_table(session):

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
