from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from word_table import Word
import os
import sys
import gensim
import nltk
import numpy as np
from global_constants import *
from helper_utils import *
from time import time

if __name__ == '__main__':

    # Parse options
    options = parse_options(sys.argv)

    # Link an engine to the database
    username = os.environ['PUN_USER_NAME']
    password = os.environ['PUN_USER_PASSWORD']
    engine = create_engine('postgresql://{}:{}@localhost/entendrepreneur_db'.format(username, password))

    # Link a session to the engine, and instantiate
    Session = sessionmaker(bind=engine)
    session = Session()

    # Continue prompting user for inputs until program is terminated
    while True:
        if not options['test']:
            # Read and validate the inputs, and return the appropriate error message if anything is wrong with them
            input_string = raw_input("\nSeed words:  ")
            status, message = validate_input(input_string, session)
            if status == 1:
                print message
                continue

            # Split input string into the two consituent graphemes
            grapheme1, grapheme2 = input_string.split()

            # Find the semantic neighbors of the graphemes
            start = time()
            nearest_graphemes1 = get_semantic_neighbor_graphemes(grapheme1, session)
            nearest_graphemes2 = get_semantic_neighbor_graphemes(grapheme2, session)
            print 'Finding nearest-neighbor graphemes: {:.2f}sec'.format(time()-start)

            # Find the Word objects corresponding to each of the semantic neighbors
            start = time()
            nearest_words1 = session.query(Word).filter(Word.grapheme.in_(nearest_graphemes1)).all()
            nearest_words2 = session.query(Word).filter(Word.grapheme.in_(nearest_graphemes1)).all()
            print 'Nearest-neighbor graphemes to words: {:.2f}sec'.format(time()-start)
        else: # if running in test mode, use pre-selected inputs
            grapheme1, grapheme2 = parse_input(TEST_INPUT)
            nearest_words1 = session.query(Word).filter(Word.grapheme==grapheme1).all()
            nearest_words2 = session.query(Word).filter(Word.grapheme==grapheme2).all()

        # Compute Portmanteaus out of pairs of semantic neighbors
        start = time()
        portmanteaus = get_portmanteaus(nearest_words1, nearest_words2, session)
        print 'Portmanteau generation: {:.2f}sec'.format(time()-start)

        # Compute Rhymes out of pairs of semantic neighbors
        start = time()
        rhymes = get_rhymes(nearest_words1, nearest_words2, session)
        print 'Rhyme generation: {:.2f}sec'.format(time()-start)

        # Print the top MAX_PORTMANTEAUS many Portmanteaus
        print '''
        ########################
        ##### PORTMANTEAUS #####
        ########################
        '''
        for i, portmanteau in enumerate(portmanteaus):
            if i >= MAX_PORTMANTEAUS:
                break
            if options['debug']: # print extra info if debug mode is on
                print repr(portmanteau)
            else:
                print portmanteau

        # Print the top MAX_RHYMES many Rhymes
        print '''
        ##################
        ##### RHYMES #####
        ##################
        '''
        for i, rhyme in enumerate(rhymes):
            if i >= MAX_RHYMES:
                break
            if options['debug']: # print extra info if debug mode is on
                print repr(rhyme)
            else:
                print rhyme

        # If it's a test run, we only want to run the while-loop once
        if options['test']:
            break
