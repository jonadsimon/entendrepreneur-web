from sqlalchemy import Column, Integer, String, Text
from base import Base

class SubgraphemeFrequency(Base):
    __tablename__ = 'subgrapheme_frequencies'

    id = Column(Integer, primary_key=True)
    grapheme = Column(Text) # needs to be Text rather than String (i.e. varchar) to avoid typing issues
    frequency = Column(Integer, default=1)
    frequency_head = Column(Integer, default=1)
    frequency_tail = Column(Integer, default=1)

    def __repr__(self):
        return "<SubgraphemeFrequency(grapheme='%s', frequency=%i, frequency_head=%i, frequency_tail=%i)>" % (self.grapheme, self.frequency, self.frequency_head, self.frequency_tail)

    @staticmethod
    def get_subgrapheme_frequency(this_grapheme, session, side='all'):
        '''
        Return the frequency of the grapheme
        '''
        try:
            subgrapheme_frequency = session.query(SubgraphemeFrequency).filter(SubgraphemeFrequency.grapheme==this_grapheme).one()
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
