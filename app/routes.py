from flask import render_template, url_for, redirect, request, session
from app import app, db
from app.models import Word, SubgraphemeFrequency, SubphonemeFrequency
from app.forms import InputWords
from app.helper_utils import get_semantic_neighbor_graphemes, get_portmanteaus, get_rhymes
from app.global_constants import MAX_PORTMANTEAUS, MAX_RHYMES
from time import time

def get_subgrapheme_frequency_cache(words):
    '''
    Generate graphene frequency cache up front, so db doesn't need to be queried ~1000x additional times
    '''
    head_subgraphemes = [word.grapheme[:i] for word in words for i in range(1, min(len(word.grapheme)+1, 6))]
    tail_subgraphemes = [word.grapheme[-i:] for word in words for i in range(1, min(len(word.grapheme)+1, 6))]
    subgraphemes = list(set(head_subgraphemes + tail_subgraphemes))
    print '# Subgraphemes:', len(subgraphemes), ', Ex:', subgraphemes[0]
    # subgrapheme_frequency_rows = SubgraphemeFrequency.query.filter(SubgraphemeFrequency.grapheme.in_(subgraphemes)).all()
    query = SubgraphemeFrequency.query.filter(SubgraphemeFrequency.grapheme.in_(subgraphemes))
    params = {'grapheme_{}'.format(i+1): grapheme for i,grapheme in enumerate(subgraphemes)}
    expl = db.engine.execute('EXPLAIN ANALYZE ' + str(query), params)
    for row in expl:
        for r in row:
            print r
    subgrapheme_frequency_rows = query.all()
    return {row.grapheme: {'head': row.frequency_head, 'tail': row.frequency_tail} for row in subgrapheme_frequency_rows}

def get_subphoneme_frequency_cache(words):
    '''
    Generate phoneme frequency cache up front, so db doesn't need to be queried ~1000x additional times
    '''
    head_subphonemes = [word.phoneme[:i] for word in words for i in range(1, min(len(word.phoneme)+1, 6))]
    tail_subphonemes = [word.phoneme[-i:] for word in words for i in range(1, min(len(word.phoneme)+1, 6))]
    subphonemes = map(list, list(set(map(tuple, head_subphonemes) + map(tuple, tail_subphonemes))))
    print '# Subphonemes:', len(subphonemes), ', Ex:', subphonemes[0]
    # subphoneme_frequency_rows = SubphonemeFrequency.query.filter(SubphonemeFrequency.phoneme.in_(subphonemes)).all()
    query = SubphonemeFrequency.query.filter(SubphonemeFrequency.phoneme.in_(subphonemes))
    params = {'phoneme_{}'.format(i+1): '{'+','.join(map(str, subphoneme))+'}' for i,subphoneme in enumerate(subphonemes)}
    expl = db.engine.execute('EXPLAIN ANALYZE ' + str(query), params)
    for row in expl:
        for r in row:
            print r
    subphoneme_frequency_rows = query.all()
    return {tuple(row.phoneme): {'head': row.frequency_head, 'tail': row.frequency_tail} for row in subphoneme_frequency_rows}

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

    # Generate subgrapheme and subphoneme frequency caches up-front
    start = time()
    subgrapheme_frequency_cache = get_subgrapheme_frequency_cache(nearest_words1 + nearest_words2)
    print "Subgrapheme frequency cache: {:.2f} seconds".format(time()-start)

    start = time()
    subphoneme_frequency_cache = get_subphoneme_frequency_cache(nearest_words1 + nearest_words2)
    print "Subphoneme frequency cache: {:.2f} seconds".format(time()-start)

    # Generate the ordered portmanteaus
    start = time()
    portmanteaus = get_portmanteaus(nearest_words1, nearest_words2, subgrapheme_frequency_cache, subphoneme_frequency_cache)
    print "Portmanteaus: {:.2f} seconds".format(time()-start)

    # Generate the ordered rhymes
    start = time()
    rhymes = get_rhymes(nearest_words1, nearest_words2, subphoneme_frequency_cache)
    print "Rhymes: {:.2f} seconds".format(time()-start)

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
