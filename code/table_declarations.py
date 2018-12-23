# will start by connecting to local postgres db before switching to AWS RDS
# can't start off on sqlite because it doesn't support json

# use table definition as previously planned BUT instead of storing grapheme_chunks/phoneme_chunks
# define a custom SequenceAlignment type as shown here: https://docs.sqlalchemy.org/en/latest/orm/composites.html
# ~~ ACTUALLY NO, DON'T WORRY ABOUT THIS PART FOR NOW ~~

# instructions for setting up a local postgres db:
# https://www.codementor.io/engineerapart/getting-started-with-postgresql-on-mac-osx-are8jcopb

# brew services start postgresql
# CREATE ROLE pun_user WITH LOGIN PASSWORD 'punsaregreat';
# CREATE DATABASE entendrepreneur_db;

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, aliased
from sqlalchemy.orm.session import object_session

from global_constants import MAX_NEIGHBORS, VOCAB_SIZE
import numpy as np
# from sequence_alignment import SequenceAlignment
# from pronunciation_dictionary import PronunciationDictionary

import os
username = os.environ['PUN_USER_NAME']
password = os.environ['PUN_USER_PASSWORD']

engine = create_engine('postgresql://{}:{}@localhost/entendrepreneur_db'.format(username, password))

Base = declarative_base()

class Word(Base): # will likely cause a name collision... "GraphemePhonemePair"
    __tablename__ = 'words'

    id = Column(Integer, primary_key=True)
    grapheme = Column(String)
    phoneme = Column(ARRAY(String))
    grapheme_chunks = Column(JSON)
    phoneme_chunks = Column(JSON)

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
    		subgrapheme_start_idx = sum(map(len, self.seq1[:start_chunk_idx]))
    		subgrapheme_end_idx = sum(map(len, self.seq1[:end_chunk_idx+1])) - 1
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
        result = session.execute(query, {'word_id': self.id, max_neighbors: MAX_NEIGHBORS}) # pass in query params

        neighbor_word_ids = [row['word_id'] for row in result]
        neighbor_words = session.query(Word).filter_by(Word.id.in_(neighbor_word_ids)).all()

        return neighbor_words

class SubgraphemeFrequency(Base):
    __tablename__ = 'subgrapheme_frequencies'

    id = Column(Integer, primary_key=True)
    grapheme = Column(String)
    frequency = Column(Integer, default=1)
    frequency_head = Column(Integer, default=1)
    frequency_tail = Column(Integer, default=1)

    def __repr__(self):
        return "<SubgraphemeFrequency(grapheme='%s', frequency=%i, frequency_head=%i, frequency_tail=%i)>" % (self.grapheme, self.frequency, self.frequency_head, self.frequency_tail)

    @staticmethod
	def get_subgrapheme_frequency(this_grapheme, side='all'):
		'''
        Return the frequency of the grapheme
		'''
        subgrapheme_frequency = SubgraphemeFrequency.query.filter(grapheme==this_grapheme).one()
		if side == 'head':
			return subgrapheme_frequency.frequency_head
		elif side == 'tail':
			return subgrapheme_frequency.frequency_tail
		elif side == 'all':
			return subgrapheme_frequency.frequency
		else:
			raise "Argument 'side' must be either 'head', 'tail', or 'all'"

# Almost identical to SubgraphemeFrequency... consider making a Parent class
class SubphonemeFrequency(Base):
    __tablename__ = 'subphoneme_frequencies'

    id = Column(Integer, primary_key=True)
    phoneme = Column(ARRAY(String))
    frequency = Column(Integer, default=1)
    frequency_head = Column(Integer, default=1)
    frequency_tail = Column(Integer, default=1)

    def __repr__(self):
        return "<SubphonemeFrequency(phoneme='%s', frequency=%i, frequency_head=%i, frequency_tail=%i)>" % (self.phoneme, self.frequency, self.frequency_head, self.frequency_tail)

    @staticmethod
	def get_subphoneme_frequency(this_phoneme, side='all'):
		'''
        Return the frequency of the phoneme
		'''
        subphoneme_frequency = SubphonemeFrequency.query.filter(phoneme==this_phoneme).one()
		if side == 'head':
			return subphoneme_frequency.frequency_head
		elif side == 'tail':
			return subphoneme_frequency.frequency_tail
		elif side == 'all':
			return subphoneme_frequency.frequency
		else:
			raise "Argument 'side' must be either 'head', 'tail', or 'all'"

class FasttextVectorElement(Base):
    __tablename__ = 'fasttext_vector_elements'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'))
    index = Column(Integer)
    value = Column(Float) # these values are NOT L2-normalized, need to do that on each call to most_similar

    word = relationship('Word', back_populates='fasttext_vector_elements')

    def __repr__(self):
        return "<FasttextVector(word_id=%i, index=%i, value=%d)>" % (self.word_id, self.index, self.value)


Word.fasttext_vector_elements = relationship('FasttextVectorElement', order_by=FasttextVectorElement.id, back_populates='word') # 'words' ?

Base.metadata.create_all(engine) # create the tables


# Populate the tables...

# Write it as a single file, then move to ipython notebook

# Delete the (many) now-irrelevant Word, PronunciationDictionary, and SubwordFrequency classes
