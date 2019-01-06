# Script is called from the top-level entendrepreneur-web directory as:
# > python manage.py populate_tables

from flask_script import Manager
from app import app, db
from app.models import *
from time import time

import sys
sys.path.insert(0, 'scripts') # need to add the code path for other imports to work
from populate_word_table import populate_word_table
from populate_subgrapheme_frequency_table import populate_subgrapheme_frequency_table
from populate_subphoneme_frequency_table import populate_subphoneme_frequency_table
from populate_fasttext_neighbor_table import populate_fasttext_neighbor_table

manager = Manager(app)

@manager.command
def populate_tables():
    '''
    Pass in the Class instantiators directly to avoid import issues
    Takes ~8min to run with the current 4 tables
    '''
    start = time()
    # Populate the Word table, and commit the changes
    populate_word_table(Word, db)
    print 'Finished populating Word table after {:.0f} seconds'.format(time()-start)

    start = time()
    # Populate the SubgraphemeFrequency table, and commit the changes
    populate_subgrapheme_frequency_table(SubgraphemeFrequency, db)
    print 'Finished populating SubgraphemeFrequency table after {:.0f} seconds'.format(time()-start)

    start = time()
    # Populate the SubphonemeFrequency table, and commit the changes
    populate_subphoneme_frequency_table(SubphonemeFrequency, db)
    print 'Finished populating SubphonemeFrequency table after {:.0f} seconds'.format(time()-start)

    start = time()
    # Populate the FasttextNeighbor table, and commit the changes
    populate_fasttext_neighbor_table(FasttextNeighbor, db)
    print 'Finished populating FasttextNeighbor table after {:.0f} seconds'.format(time()-start)

if __name__ == "__main__":
    manager.run()
