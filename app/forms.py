from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import ValidationError, DataRequired
from app.helper_utils import alternate_capitalizations
from app.models import FasttextNeighbor

class InputWords(FlaskForm):
    word1 = StringField('word 1', validators=[DataRequired()])
    word2 = StringField('word 2', validators=[DataRequired()])
    submit = SubmitField('Generate Puns')

    # Annoying that the validation is identical/redundant, but keep as-is for now
    # Will return double-errors if both words are identical AND unrecognized
    def validate_word1(self, word1):
        word1_alternates = alternate_capitalizations(word1.data)
        word1_grapheme = FasttextNeighbor.query.filter(FasttextNeighbor.grapheme.in_(word1_alternates)).first()
        if word1_grapheme is None:
            raise ValidationError('Word not recognized, check spelling and capitalization.')

    def validate_word2(self, word2):
        word2_alternates = alternate_capitalizations(word2.data)
        word2_grapheme = FasttextNeighbor.query.filter(FasttextNeighbor.grapheme.in_(word2_alternates)).first()
        if word2_grapheme is None:
            raise ValidationError('Word not recognized, check spelling and capitalization.')
