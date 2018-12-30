from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

# Read postgres username and password from the OS environment
import os
username = os.environ['PUN_USER_NAME']
password = os.environ['PUN_USER_PASSWORD']

# Link an engine to the database
engine = create_engine('postgresql://{}:{}@localhost/entendrepreneur_db'.format(username, password))
Base = declarative_base(bind=engine)

# Drop all tables in the database
Base.metadata.drop_all(bind=engine)
print 'Finished dropping all tables'
