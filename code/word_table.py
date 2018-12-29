from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ARRAY, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.orm.session import object_session

from global_constants import MAX_NEIGHBORS
import numpy as np

Base = declarative_base()

class Word(Base): # will likely cause a name collision... "GraphemePhonemePair"
    __tablename__ = 'words'

    id = Column(Integer, primary_key=True)
    grapheme = Column(String)
    phoneme = Column(ARRAY(String))
    grapheme_chunks = Column(JSON)
    phoneme_chunks = Column(JSON)

    fasttext_vector_elements = relationship("FasttextVectorElement")

    def __repr__(self):
        return "<Word(grapheme='%s', phoneme='%s')>" % (self.grapheme, '-'.join(self.phoneme))

    def get_subgrapheme_from_subphoneme_inds(self, start_idx, end_idx, return_inds=False):
        '''
        Return the subgrapheme string (return_inds=False), or inclusive subgrapheme indices (return_inds=True),
        corresponding to the subphoneme starting at start_idx and ending at end_idx, inclusive
        '''
		chunk_lengths = map(len, self.phoneme_chunks)
		valid_end_inds = np.cumsum(chunk_lengths) - 1
		valid_start_inds = np.cumsum(chunk_lengths) - chunk_lengths

		if start_idx not in valid_start_inds:
			raise Exception('\'start_idx\' falls in the middle of a phoneme chunk')

		if end_idx not in valid_end_inds:
			raise Exception('\'end_idx\' falls in the middle of a phoneme chunk')

		# Include null-graphs at the boundaries
		start_chunk_idx = np.where(valid_start_inds == start_idx)[0].min()
		end_chunk_idx = np.where(valid_end_inds == end_idx)[0].max()

        if not return_inds:
            # Return the subgrapheme corresponding to the providing subphoneme indices
            subgrapheme = sum(map(list, self.grapheme_chunks[start_chunk_idx:end_chunk_idx+1]), [])
            return subgrapheme
        else:
            # Return the subgrapheme *indices* corresponding to the providing subphoneme indices
    		subgrapheme_start_idx = sum(map(len, self.grapheme_chunks[:start_chunk_idx]))
    		subgrapheme_end_idx = sum(map(len, self.grapheme_chunks[:end_chunk_idx+1])) - 1
    		return subgrapheme_start_idx, subgrapheme_end_idx

    def get_destressed_phoneme(self):
        '''
        Strip the stress information off of the phoneme
        '''
        return [filter(str.isalpha, str(phone)) for phone in self.phoneme]

    def get_semantic_neighbors(self):
        '''
        Computes the cosine distance of word with every other word vector in FasttextVectorElement,
        and returns the nearest MAX_NEIGHBORS many words, along with word itself

        Very difficult to structure this query in SQLAlchemy's query meta-language,
        so just use 'execute' to run the raw SQL
        '''

        session = object_session(self) # get the session that the current Word instance instance is associated with

        query = '''
        SELECT
            fv2.word_id,
            SUM(fv1.value * fv2.value) / (SQRT(SUM(fv1.value * fv1.value)) * SQRT(SUM(fv2.value * fv2.value))) cosine_dist
        FROM fasttext_vector_elements fv1
        JOIN fasttext_vector_elements fv2
        ON
            fv1.word_id = :word_id
            AND fv1.index = fv2.index
        GROUP BY 1
        ORDER BY 2
        LIMIT :max_neighbors + 1
        '''
        result = session.execute(query, {'word_id': self.id, 'max_neighbors': MAX_NEIGHBORS}) # pass in query params

        neighbor_word_ids = [row['word_id'] for row in result]
        neighbor_words = session.query(Word).filter_by(Word.id.in_(neighbor_word_ids)).all()

        return neighbor_words
