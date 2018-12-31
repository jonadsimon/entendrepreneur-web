from time import time
import gensim
import sys
sys.path.insert(0, '../code')
from global_constants import REPO_HOME, MAX_VOCAB
from fasttext_vector_table import FasttextVector

def populate_fasttext_vector_table(session):
    # Load the word vectors
    start = time()
    fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(REPO_HOME+'data/word_vectors/wiki-news-300d-1M.vec', limit=MAX_VOCAB)
    print 'Finished loading fasttext_model into memory: {:.0f} seconds'.format(time()-start)

    # Create a FasttextVector object for each word vector
    start = time()
    fasttext_vector_list = []
    for grapheme_idx, grapheme in enumerate(fasttext_model.vocab.iterkeys()):
        new_fasttext_vector_args = {'grapheme': grapheme}

        # Will only be using vectors to compute dot products, so normalize in advance to save on computation
        normalized_unit_vec = gensim.matutils.unitvec(fasttext_model.get_vector(grapheme)).astype(float)
        new_fasttext_vector_args.update({'v{}'.format(i+1) : normalized_unit_vec[i] for i in range(300)})
        new_fasttext_vector = FasttextVector(**new_fasttext_vector_args)
        fasttext_vector_list.append(new_fasttext_vector)

        # Running into out-of-memory errors storing so many objects in memory,
        # so dump the objects into the db every 1000 vectors, and clear the accumulated cache
        if grapheme_idx % 1000 == 0 and grapheme_idx > 0:
            print 'Finished processing vector {}'.format(grapheme_idx)
            session.add_all(fasttext_vector_list)
            session.commit()
            fasttext_vector_list = []
            print 'Finished committing vector {}'.format(grapheme_idx)
            print 'Looping duration elapsed: {:.0f} seconds'.format(time()-start)
