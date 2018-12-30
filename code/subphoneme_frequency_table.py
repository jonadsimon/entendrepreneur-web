from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ARRAY
from base import Base

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
