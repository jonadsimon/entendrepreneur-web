from sqlalchemy import create_engine

import sys
sys.path.insert(0, '../app') # need to add the code path for other imports to work
# Import tables up front so that table deletion works correctly
from base import Base
from word_table import Word
from subgrapheme_frequency_table import SubgraphemeFrequency
from subphoneme_frequency_table import SubphonemeFrequency
from fasttext_vector_table import FasttextVector
from fasttext_neighbor_table import FasttextNeighbor

# Read postgres username and password from the OS environment
import os
username = os.environ['PUN_USER_NAME']
password = os.environ['PUN_USER_PASSWORD']

# Link an engine to the database
engine = create_engine('postgresql://{}:{}@localhost/entendrepreneur_db'.format(username, password))

# Drop all tables in the database
Base.metadata.drop_all(bind=engine)
print 'Finished dropping all tables'
