import gensim
import numpy as np
import sys
sys.path.insert(0, '../app')
from global_constants import REPO_HOME, MAX_VOCAB
from sklearn.decomposition import PCA
from time import time

def post_processing_algorithm(X, D):
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

def dimensionality_reduction_algorithm(X, D=7):
    '''
    Reduce word vector dimension from 300 to 150. Use D=7 as recommended in the paper

    From "Simple and Effective Dimensionality Reduction for Word Embeddings"
    '''
    start = time()
    X = post_processing_algorithm(X, D)
    print 'Finished PPA #1: {:.0f} seconds'.format(time()-start)

    start = time()
    pca = PCA(n_components=150, svd_solver='arpack') # need to manually set svd_solver to avoid memory issues
    X = pca.fit_transform(X)
    print 'Finished PCA-150: {:.0f} seconds'.format(time()-start)

    start = time()
    X = post_processing_algorithm(X, D)
    print 'Finished PPA #2: {:.0f} seconds'.format(time()-start)

    return X


# Load the word vectors
# Only care about the first 300k vectors, ignore the rest for now (for memory issue reasons)
start = time()
fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(REPO_HOME+'data/word_vectors/wiki-news-300d-1M.vec', limit=MAX_VOCAB)
print 'Finished loading fasttext_model into memory: {:.0f} seconds'.format(time()-start)

fasttext_model.vectors = dimensionality_reduction_algorithm(fasttext_model.vectors)

# Save this dimensionality-reduced model for later loading and exploration
start = time()
fasttext_model.save_word2vec_format(REPO_HOME+'data/word_vectors/wiki-news-150d-300K.vec')
print 'Finished saving dimensionality-reduced fasttext_model to disk: {:.0f} seconds'.format(time()-start)
