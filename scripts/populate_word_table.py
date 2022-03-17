import numpy as np
from nltk.corpus import cmudict
from app.global_constants import REPO_HOME

# Aligned pairs use the following syntax:
# 1) chunked graphemes/phonemes are divided by '|' symbols
# 2) two graphemes/phonemes which are chunked together in a mapping will be separated by a ':'
# 3) graphemes mapping to null-phonemes are denoted by '_'
DIVIDER_CHAR = '|'
CONCAT_CHAR = ':'
NULL_CHAR = '_' # null char can also appear in the grapheme sequence, but NOT as a null-graph

# CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

def m2m_grapheme_to_grapheme_chunks(m2m_grapheme):
    '''
    Convert from an m2m-aligned grapheme represention to a tuple-based grapheme represention
    e.g. 'i|m|p|e|l:l|e|d|' --> [('i',),('m',),('p',),('e',),('l','l',),('e',),('d',)]
    '''
    grapheme_chunks = m2m_grapheme.strip(DIVIDER_CHAR).split(DIVIDER_CHAR)
    new_grapheme_chunks = []
    for chunk in grapheme_chunks:
        # do NOT filter out null_chars
        new_chunk = tuple(chunk.split(CONCAT_CHAR))
        new_grapheme_chunks.append(new_chunk)
    return new_grapheme_chunks

def m2m_phoneme_to_phoneme_chunks(m2m_phoneme):
    '''
    Convert from an m2m-aligned phoneme represention to a tuple-based phoneme represention
    e.g. 'IH|M|P|EH|L|_|D|' --> [('IH',),('M',),('P',),('EH',),('L',),('_',),('D',)]
    '''
    phoneme_chunks = m2m_phoneme.strip(DIVIDER_CHAR).split(DIVIDER_CHAR)
    new_phoneme_chunks = []
    for chunk in phoneme_chunks:
        # DO filter out null_chars, here denoting silent-letters
        if chunk == NULL_CHAR:
            new_chunk = ()
        else:
            new_chunk = tuple(chunk.split(CONCAT_CHAR))
        new_phoneme_chunks.append(new_chunk)
    return new_phoneme_chunks

def grapheme_chunks_to_grapheme_string(grapheme_chunks):
    '''
    Convert from a tuple-based grapheme represention to a string-based grapheme represention
    e.g. [('i',),('m',),('p',),('e',),('l','l',),('e',),('d',)] --> 'impelled'
    '''
    return ''.join(sum(list(map(list, grapheme_chunks)), []))

def phoneme_chunks_to_stressed_phoneme_chunks(phoneme_chunks, grapheme):
    '''
    Convert from stressless phonememe represention to a stressed phoneme represention
    e.g. [('IH',),('M',),('P',),('EH',),('L',),('_',),('D',)] --> [('IH0',),('M',),('P',),('EH1',),('L',),('_',),('D',)]
    '''
    chunk_lengths = list(map(len, phoneme_chunks))
    valid_end_inds = np.cumsum(chunk_lengths)
    valid_start_inds = np.cumsum(chunk_lengths) - chunk_lengths
    idx_pairs = list(zip(valid_start_inds,valid_end_inds))

    stressed_phoneme = cmu_dict[grapheme][0]
    stressed_phoneme_chunks = [tuple(stressed_phoneme[start_idx:end_idx]) for (start_idx,end_idx) in idx_pairs]
    return stressed_phoneme_chunks, stressed_phoneme

def populate_word_table(Word, db):
    '''
    Take the current db session as an argument, and populate the words table
    '''

    # Load the aligned grapheme/phoneme pairs
    with open(REPO_HOME+'data/g2p_alignment/m2m_preprocessed_cmudict.txt.m-mAlign.2-2.delX.1-best.conYX.align') as infile:
        aligned_grapheme_phoneme_pairs = [line.strip().split('\t') for line in infile.readlines()]

    # Transform the aligned grapheme/phoneme pairs to conform to the Word table schema
    word_list = []
    for m2m_grapheme, m2m_phoneme in aligned_grapheme_phoneme_pairs:
        grapheme_chunks = m2m_grapheme_to_grapheme_chunks(m2m_grapheme)
        grapheme = grapheme_chunks_to_grapheme_string(grapheme_chunks)

        phoneme_chunks = m2m_phoneme_to_phoneme_chunks(m2m_phoneme)
        stressed_phoneme_chunks, stressed_phoneme = phoneme_chunks_to_stressed_phoneme_chunks(phoneme_chunks, grapheme)

        new_word = Word(grapheme=grapheme, phoneme=cmu_dict[grapheme][0], grapheme_chunks=grapheme_chunks, phoneme_chunks=stressed_phoneme_chunks)
        word_list.append(new_word)

    # Add the generated Word objects to the Word table, and commit the changes
    db.session.add_all(word_list)
    db.session.commit()
