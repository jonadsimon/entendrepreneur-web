# will start by connecting to local postgres db before switching to AWS RDS
# can't start off on sqlite because it doesn't support json

# instructions for setting up a local postgres db:
# https://www.codementor.io/engineerapart/getting-started-with-postgresql-on-mac-osx-are8jcopb
#
# brew services start postgresql
# CREATE ROLE pun_user WITH LOGIN PASSWORD 'punsaregreat';
# CREATE DATABASE entendrepreneur_db;

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from populate_word_table import populate_word_table
from populate_subgrapheme_frequency_table import populate_subgrapheme_frequency_table
from populate_subphoneme_frequency_table import populate_subphoneme_frequency_table
from populate_fasttext_vector_element_table import populate_fasttext_vector_element_table

# Read postgres username and password from the OS environment
import os
username = os.environ['PUN_USER_NAME']
password = os.environ['PUN_USER_PASSWORD']

# Link an engine to the database
engine = create_engine('postgresql://{}:{}@localhost/entendrepreneur_db'.format(username, password))

# Load the base class, and use it to create the data tables
Base = declarative_base()
Base.metadata.create_all(engine) # create the tables

# Link a session to the engine, and instantiate
Session = sessionmaker(bind=engine)
session = Session()


# Populate the Word table, and commit the changes
populate_word_table(session)

# Populate the SubgraphemeFrequency table, and commit the changes
populate_subgrapheme_frequency_table(session)

# Populate the SubphonemeFrequency table, and commit the changes
populate_subphoneme_frequency_table(session)

# Populate the FasttextVectorElement table, and commit the changes
populate_fasttext_vector_element_table(session)
