import gensim
import sys
sys.path.insert(0, '../code')
from global_constants import REPO_HOME, MAX_VOCAB
from fasttext_grapheme_table import FasttextGrapheme

def populate_fasttext_grapheme_table(session):
    # Load the word vectors, and create a FasttextGrapheme object for each grapheme
    # Manually set the id so that it can be easily replicated in FasttextVectorElement
    fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(REPO_HOME+'data/word_vectors/wiki-news-300d-1M.vec', limit=MAX_VOCAB)
    grapheme_list = [FasttextGrapheme(id=grapheme_idx+1, grapheme=grapheme) for (grapheme_idx, grapheme) in enumerate(fasttext_model.vocab.iterkeys())]

    # Add the generated FasttextGrapheme objects to the FasttextGrapheme table, and commit the changes
    session.add_all(grapheme_list)
    session.commit()
