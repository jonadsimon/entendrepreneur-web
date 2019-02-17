from flask import render_template, url_for, redirect, request, session
from app import app, db
from app.models import Word, UserInput
from app.forms import InputWords
from app.helper_utils import get_semantic_neighbor_graphemes, get_portmanteaus, get_rhymes
from app.global_constants import MAX_PORTMANTEAUS, MAX_RHYMES
from time import time
from datetime import datetime

def get_puns_from_words(word1, word2):
    # Find the semantic neighbors of the graphemes
    start = time()
    nearest_graphemes1 = get_semantic_neighbor_graphemes(word1)
    nearest_graphemes2 = get_semantic_neighbor_graphemes(word2)
    print "Semantic neighbors: {:.2f} seconds".format(time()-start)

    # Find the Word objects corresponding to each of the semantic neighbors
    start = time()
    nearest_words1 = Word.query.filter(Word.grapheme.in_(nearest_graphemes1)).all()
    nearest_words2 = Word.query.filter(Word.grapheme.in_(nearest_graphemes2)).all()
    print "Word conversion: {:.2f} seconds".format(time()-start)

    # Generate the ordered portmanteaus
    start = time()
    portmanteaus = get_portmanteaus(nearest_words1, nearest_words2)
    print "Portmanteaus: {:.2f} seconds".format(time()-start)

    # Generate the ordered rhymes
    start = time()
    rhymes = get_rhymes(nearest_words1, nearest_words2)
    print "Rhymes: {:.2f} seconds".format(time()-start)

    return {'portmanteaus': map(lambda x: x.serialize(), portmanteaus[:MAX_PORTMANTEAUS]), 'rhymes': map(lambda x: x.serialize(), rhymes[:MAX_RHYMES])}

def log_user_inputs(grapheme1, grapheme2, is_valid):
    '''
    Log the user's inputs
    IP address logic quelled from this thread: https://stackoverflow.com/questions/3759981/get-ip-address-of-visitors-using-flask-for-python
    '''
    ts = datetime.utcnow()
    user_ip = request.environ.get('HTTP_X_REAL_IP') or request.environ.get('REMOTE_ADDR') or request.remote_addr
    user_inputs = UserInput(grapheme1=grapheme1, grapheme2=grapheme2, is_valid=is_valid, ip_address=user_ip, created_at=ts, updated_at=ts)
    db.session.add(user_inputs)
    db.session.commit()

@app.route('/')
def home():
    '''
    Always redirect from 'home' to 'pun_generator' since there's nothing else there
    '''
    return redirect(url_for('pun_generator'))

@app.route('/pun_generator', methods=['GET', 'POST'])
def pun_generator():
    '''
    If user inputs invalid word(s), display error(s)
    Otherwise redirect to results page, passing along the inputs
    '''
    # Get the user inputs from the form
    form = InputWords()
    if form.validate_on_submit():
        # Add user inputs to the user_inputs table
        log_user_inputs(form.word1.data, form.word2.data, True)
        # Redirect to the results page, passing along the (valid) input words
        return redirect(url_for('results', word1=form.word1.data, word2=form.word2.data))

    # Submit button was clicked, but words were invalid
    if form.word1.data is not None and form.word2.data is not None: # submit button was pressed, but the inputs were invalid
        log_user_inputs(form.word1.data, form.word2.data, False)

    # If the submit button was not clicked, or the results were not valid, render the page as-is
    return render_template('pun_generator.html', form=form)

@app.route('/pun_generator/<word1>+<word2>', methods=['GET', 'POST'])
def results(word1, word2):
    '''
    Display the results, and continue prompting for input
    Assumes that the <word1>, <word2> fields in the URL are valid words, displays "Internal Server Error" if they're not
    '''
    # Get the user inputs from the form
    form = InputWords()
    if form.validate_on_submit(): # Option (1) user just inputed a new valid word pair
        # Add user inputs to the user_inputs table
        log_user_inputs(form.word1.data, form.word2.data, True)
        # Redirect to new results page with appropriate url path
        return redirect(url_for('results', word1=form.word1.data, word2=form.word2.data))
    elif form.word1.data is None and form.word2.data is None: # Option (2) user just landed on this url
        # Generate the puns, impute the form data, and display the results
        pun_results = get_puns_from_words(word1, word2)
        form.word1.data, form.word2.data = word1, word2
        return render_template('pun_generator.html', form=form, results=pun_results)
    else: # Option (3) user just inputed a new invalid word pair
        # Log the invalid words
        log_user_inputs(form.word1.data, form.word2.data, False)
        # Render the page as-is, with the errors shown
        return render_template('pun_generator.html', form=form)
