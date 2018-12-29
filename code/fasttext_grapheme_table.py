from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, ForeignKey

Base = declarative_base()

class FasttextGrapheme(Base):
    __tablename__ = 'fasttext_graphemes'

    id = Column(Integer, primary_key=True)
    grapheme = Column(String)

    fasttext_vector_elements = relationship("FasttextVectorElement")

    def __repr__(self):
        return "<FasttextGrapheme(grapheme=%s)>" % (self.grapheme, self.index, self.value)
