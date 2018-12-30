from time import time
import gensim
import sys
sys.path.insert(0, '../code')
from global_constants import REPO_HOME, MAX_VOCAB
from fasttext_vector_tables import FasttextVectorElement

def populate_fasttext_vector_element_table(session):
    # Load the word vectors
    start = time()
    fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(REPO_HOME+'data/word_vectors/wiki-news-300d-1M.vec', limit=MAX_VOCAB)
    print 'Finished loading fasttext_model into memory: {:.0f} seconds'.format(time()-start)

    # Create a FasttextVectorElement object for each element of each word vector
    # Use the same convention for setting grapheme_id as in populate_fasttext_grapheme_table.py
    start = time()
    vector_element_list = []
    for grapheme_idx, grapheme in enumerate(fasttext_model.vocab.iterkeys()):
        for vector_element_idx, vector_element in enumerate(fasttext_model.vectors[grapheme_idx]):
            # Make sure to convert the np.float32 to a standard float before object instantiation
            new_vector_element = FasttextVectorElement(grapheme_id=grapheme_idx+1, index=vector_element_idx, value=vector_element.astype(float))
            vector_element_list.append(new_vector_element)
        # Running into out-of-memory errors storing so many objects in memory,
        # so dump the objects into the db every 1000 vectors, and clear the accumulated cache
        if grapheme_idx % 1000 == 0 and grapheme_idx > 0:
            print 'Finished processing vector {}'.format(grapheme_idx)
            session.add_all(vector_element_list)
            session.commit()
            vector_element_list = []
            print 'Finished committing vector {}'.format(grapheme_idx)
            print 'Looping duration elapsed: {:.0f} seconds'.format(time()-start)
    #
    # # Add the generated FasttextVectorElement objects to the FasttextVectorElement table, and commit the changes
    # session.add_all(vector_element_list)
    # session.commit()
