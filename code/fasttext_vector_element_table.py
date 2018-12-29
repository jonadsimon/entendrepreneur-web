from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, ForeignKey

Base = declarative_base()

class FasttextVectorElement(Base):
    __tablename__ = 'fasttext_vector_elements'

    id = Column(Integer, primary_key=True)
    grapheme_id = Column(Integer, ForeignKey('fasttext_graphemes.id'))
    index = Column(Integer)
    value = Column(Float) # these values are NOT L2-normalized, need to do that on each call to most_similar

    def __repr__(self):
        return "<FasttextVectorElement(grapheme_id=%i, index=%i, value=%d)>" % (self.grapheme_id, self.index, self.value)
