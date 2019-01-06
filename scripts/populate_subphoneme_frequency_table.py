from nltk.corpus import cmudict
from collections import defaultdict

# CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

def populate_subphoneme_frequency_table(SubphonemeFrequency, db):

    # Counters for storing the frequencies of each subphoneme
    # Smooth by using default value of 1 to avoid probability singularities when string is not present
    subphoneme_head_counts = defaultdict(lambda: 1)
    subphoneme_tail_counts = defaultdict(lambda: 1)
    subphoneme_counts = defaultdict(lambda: 1)

    for phonemes in cmu_dict.itervalues():
        # Only store subphonemes up to a length of 5
        # Anything longer than that is rare enough that the default count of 1 is a good approximation
        phoneme = tuple(phonemes[0])
        for k in range(1,6):
            for i in range(len(phoneme)-k+1):
                p = phoneme[i:i+k]
                subphoneme_counts[p] += 1
                if i == 0: # head
                    subphoneme_head_counts[p] += 1
                if i + k == len(phoneme): # tail
                    subphoneme_tail_counts[p] += 1

    # Now iterate through the subphonemes, and add them to the SubphonemeFrequency table
    subphoneme_frequency_list = []
    for p in subphoneme_counts.iterkeys():
        new_subphoneme_frequency = SubphonemeFrequency(phoneme=p,
                                                        frequency=subphoneme_counts[p],
                                                        frequency_head=subphoneme_head_counts[p],
                                                        frequency_tail=subphoneme_tail_counts[p])
        subphoneme_frequency_list.append(new_subphoneme_frequency)

    # Add the generated SubphonemeFrequency objects to the SubphonemeFrequency table, and commit the changes
    db.session.add_all(subphoneme_frequency_list)
    db.session.commit()
