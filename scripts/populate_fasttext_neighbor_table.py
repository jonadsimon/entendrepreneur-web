from time import time
import gensim
from app.global_constants import REPO_HOME
import pickle as pkl

def populate_fasttext_neighbor_table(FasttextNeighbor, db):
    # Load the precomputed grapheme neighbors
    start = time()
    grapheme_neighbor_dict = pkl.load(open(REPO_HOME+'data/word_vectors/top200_neighbors_sim35_300k.pkl', 'rb'))
    print('Finished loading fasttext_neighbors into memory: {:.0f} seconds'.format(time()-start))

    # Create a FasttextNeighbor object for each grapheme
    start = time()
    fasttext_neighbor_list = []
    for grapheme_idx, (grapheme, neighbors) in enumerate(grapheme_neighbor_dict.items()):
        new_fasttext_neighbor = FasttextNeighbor(grapheme=grapheme, neighbors=neighbors)
        fasttext_neighbor_list.append(new_fasttext_neighbor)
        # Running into out-of-memory errors storing so many objects in memory,
        # so dump the objects into the db every 50000 graphemes, and clear the accumulated cache
        if (grapheme_idx+1) % 50000 == 0:
            print('Finished processing grapheme {}'.format(grapheme_idx+1))
            db.session.add_all(fasttext_neighbor_list)
            db.session.commit()
            fasttext_neighbor_list = []
            print('Finished committing grapheme {}'.format(grapheme_idx+1))
            print('Looping duration elapsed: {:.0f} seconds'.format(time()-start))
