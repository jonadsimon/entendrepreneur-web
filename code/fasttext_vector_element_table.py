from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, ForeignKey

Base = declarative_base()

class FasttextVectorElement(Base):
    __tablename__ = 'fasttext_vector_elements'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'))
    index = Column(Integer)
    value = Column(Float) # these values are NOT L2-normalized, need to do that on each call to most_similar

    def __repr__(self):
        return "<FasttextVector(word_id=%i, index=%i, value=%d)>" % (self.word_id, self.index, self.value)
