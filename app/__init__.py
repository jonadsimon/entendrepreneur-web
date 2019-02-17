from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_sslify import SSLify
from flask_bootstrap import Bootstrap
from flask_heroku import Heroku


app = Flask(__name__)
app.config.from_object(Config)
Bootstrap(app)
sslify = SSLify(app)
heroku = Heroku(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

from app import routes, models
