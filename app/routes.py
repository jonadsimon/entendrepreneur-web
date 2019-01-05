from flask import render_template, url_for, redirect, request, session
from app import app
from app.models import Word
from app.forms import InputWords
from app.helper_utils import get_semantic_neighbor_graphemes, get_portmanteaus, get_rhymes
from app.global_constants import MAX_PORTMANTEAUS, MAX_RHYMES
from time import time

def get_puns_from_form_data(form):
    # Find the semantic neighbors of the graphemes
    start = time()
    nearest_graphemes1 = get_semantic_neighbor_graphemes(form.word1.data)
    nearest_graphemes2 = get_semantic_neighbor_graphemes(form.word2.data)
    print "Semantic neighbors: {:.2f} seconds".format(time()-start)

    # Find the Word objects corresponding to each of the semantic neighbors
    start = time()
    nearest_words1 = Word.query.filter(Word.grapheme.in_(nearest_graphemes1)).all()
    nearest_words2 = Word.query.filter(Word.grapheme.in_(nearest_graphemes2)).all()
    print "Word conversion: {:.2f} seconds".format(time()-start)

    # Generate the ordered portmanteaus and rhymes
    start = time()
    portmanteaus = get_portmanteaus(nearest_words1, nearest_words2)
    rhymes = get_rhymes(nearest_words1, nearest_words2)
    print "Portmanteaus & Rhymes: {:.2f} seconds".format(time()-start)

    return {'portmanteaus': map(lambda x: x.__str__(), portmanteaus[:MAX_PORTMANTEAUS]), 'rhymes': map(lambda x: x.__str__(), rhymes[:MAX_RHYMES])}

@app.route('/')
def home():
    '''
    Always redirect from 'home' to 'pun_generator' since there's nothing else there
    '''
    return redirect(url_for('pun_generator'))

@app.route('/pun_generator', methods=['GET', 'POST'])
def pun_generator():
    '''
    Display the results, and continue prompting for input
    '''
    # Get the user inputs from the form
    form = InputWords()
    if form.validate_on_submit():
        # If the inputs are valid, compute and store the puns
        session['results'] = get_puns_from_form_data(form)
        # rerender the page with the pun results
        return render_template('pun_generator.html', form=form, results=session['results'])
    # If the submit button was not clicked, or the results were not valid, render the page as-is
    return render_template('pun_generator.html', form=form)
