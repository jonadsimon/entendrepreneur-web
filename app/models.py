from app import db
import numpy as np


class Word(db.Model):
    __tablename__ = 'words'

    id = db.Column(db.Integer, primary_key=True)
    grapheme = db.Column(db.String)
    phoneme = db.Column(db.ARRAY(db.String))
    grapheme_chunks = db.Column(db.JSON)
    phoneme_chunks = db.Column(db.JSON)

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


class SubgraphemeFrequency(db.Model):
    __tablename__ = 'subgrapheme_frequencies'

    id = db.Column(db.Integer, primary_key=True)
    grapheme = db.Column(db.Text, index=True, unique=True) # needs to be db.Text rather than db.String (i.e. varchar) to avoid typing issues
    frequency = db.Column(db.Integer, default=1)
    frequency_head = db.Column(db.Integer, default=1)
    frequency_tail = db.Column(db.Integer, default=1)

    def __repr__(self):
        return "<SubgraphemeFrequency(grapheme='%s', frequency=%i, frequency_head=%i, frequency_tail=%i)>" % (self.grapheme, self.frequency, self.frequency_head, self.frequency_tail)

    @classmethod
    def get_subgrapheme_frequency(cls, this_grapheme, side='all'):
        '''
        Return the frequency of the grapheme
        '''
        try:
            subgrapheme_frequency = cls.query.filter_by(grapheme=this_grapheme).one()
        except:
            # if the grapheme is not present in the database i.e. because it is too long, return the default frequency of 1
            return 1
        if side == 'head':
            return subgrapheme_frequency.frequency_head
        elif side == 'tail':
            return subgrapheme_frequency.frequency_tail
        elif side == 'all':
            return subgrapheme_frequency.frequency
        else:
            raise "Argument 'side' must be either 'head', 'tail', or 'all'"


class SubphonemeFrequency(db.Model):
    __tablename__ = 'subphoneme_frequencies'

    id = db.Column(db.Integer, primary_key=True)
    phoneme = db.Column(db.ARRAY(db.Text), index=True, unique=True) # needs to be db.Text rather than db.String (i.e. varchar) to avoid typing issues
    frequency = db.Column(db.Integer, default=1)
    frequency_head = db.Column(db.Integer, default=1)
    frequency_tail = db.Column(db.Integer, default=1)

    def __repr__(self):
        return "<SubphonemeFrequency(phoneme='%s', frequency=%i, frequency_head=%i, frequency_tail=%i)>" % (self.phoneme, self.frequency, self.frequency_head, self.frequency_tail)

    @classmethod
    def get_subphoneme_frequency(cls, this_phoneme, side='all'):
        '''
        Return the frequency of the phoneme
        '''
        try:
            subphoneme_frequency = cls.query.filter_by(phoneme=this_phoneme).one()
        except:
            # if the phoneme is not present in the database i.e. because it is too long, return the default frequency of 1
            return 1
        if side == 'head':
            return subphoneme_frequency.frequency_head
        elif side == 'tail':
            return subphoneme_frequency.frequency_tail
        elif side == 'all':
            return subphoneme_frequency.frequency
        else:
            raise "Argument 'side' must be either 'head', 'tail', or 'all'"


class FasttextNeighbor(db.Model):
    __tablename__ = 'fasttext_neighbors'

    id = db.Column(db.Integer, primary_key=True)
    grapheme = db.Column(db.String, index=True, unique=True)
    neighbors = db.Column(db.ARRAY(db.String))

    def __repr__(self):
        return "<FasttextNeighbor(grapheme='%s', neighbors=...)>" % (self.grapheme)
