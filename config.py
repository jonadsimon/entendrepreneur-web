import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = os.environ.get('PUN_SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.environ.get('PUN_DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    TEMPLATES_AUTO_RELOAD = True
