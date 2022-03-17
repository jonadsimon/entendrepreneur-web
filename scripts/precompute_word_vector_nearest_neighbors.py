import gensim
import numpy as np
import sys
sys.path.insert(0, '../app')
from sklearn.decomposition import PCA
from time import time
import pickle as pkl
from app.global_constants import REPO_HOME, DEBUG

MAX_VOCAB = 300000
def post_processing_algorithm(X, D=3):
    '''
    X - n_words x n_vector_dims
    D - dimensions to cutoff

    From "ALL-BUT-THE-TOP: SIMPLE AND EFFECTIVE POSTPROCESSING FOR WORD REPRESENTATIONS"
    '''
    X_centered = X - X.mean(axis=0)
    pca = PCA(n_components=D)
    pca.fit(X_centered)
    U = pca.components_
    Y = U.dot(X.T) # unclear if this is supposed to be X or X_centered...
    Z = np.zeros(X.shape)
    for i in range(Z.shape[0]):
        Z[i] = (Y[:,i,np.newaxis]*U).sum(axis=0)
    X_prime = X_centered - Z
    return X_prime

def nearest_neighbor(grapheme, fasttext_model, n=200, s=0.35):
    '''
    Get the semantic neighbors of the grapheme.
    The cutoff for what constitutes a neighbor is either:
    1) further away than n=100 other graphemes, or
    2) <s=0.35 similarity between the two vectors
    whichever comes first

    The vectors in fasttext_model have already been normalized

    Code adapted from https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py
    '''
    sims = fasttext_model.vectors.dot(fasttext_model.get_vector(grapheme))
    topn_sim_inds = gensim.matutils.argsort(sims, topn=n+1, reverse=True)
    neighbors_with_sims = [(fasttext_model.index2word[idx], sims[idx]) for idx in topn_sim_inds]
    return [neighbor for neighbor, sim in neighbors_with_sims if sim >= s]

def precompute_word_vector_nearest_neighbors():
  # Load the word vectors
  # Only care about the first 300k vectors, ignore the rest for now (for memory issue reasons)
  start = time()
  fasttext_model = gensim.models.KeyedVectors.load_word2vec_format('data/word_vectors/wiki-news-300d-1M.vec', limit=MAX_VOCAB)
  print('Finished loading fasttext_model into memory: {:.0f} seconds'.format(time()-start))

  start = time()
  # Run the post-processing algorithm to improve the quality of the word-vectors before computing neighbors
  fasttext_model.vectors = post_processing_algorithm(fasttext_model.vectors)
  print('Finished post-processing word vectors: {:.0f} seconds'.format(time()-start))

  start = time()
  # Normalize the word-vectors to speed up the inner-product computations
  fasttext_model.vectors = fasttext_model.vectors / np.linalg.norm(fasttext_model.vectors, axis=1)[:,np.newaxis]
  print('Finished normalizing word vectors: {:.0f} seconds'.format(time()-start))
  print('Sanity-check: {} = (300000, 300)'.format(fasttext_model.vectors.shape))
  print('Sanity-check: {} = 1.0'.format(fasttext_model.vectors[65465].dot(fasttext_model.vectors[65465])))

  # Create a FasttextVector object for each word vector
  start = time()
  grapheme_neighbor_dict = {}
  for grapheme_idx, grapheme in enumerate(fasttext_model.vocab.keys()):
      neighbors = nearest_neighbor(grapheme, fasttext_model)
      grapheme_neighbor_dict.update({grapheme: neighbors})
      if grapheme_idx % 1000 == 0 and grapheme_idx > 0:
          print('Finished processing vector {}'.format(grapheme_idx))
          print('Looping duration elapsed: {:.0f} seconds'.format(time()-start))
          if DEBUG:
            break

  pkl.dump(grapheme_neighbor_dict, open(REPO_HOME + 'data/word_vectors/top200_neighbors_sim35_300k.pkl', "wb"))
