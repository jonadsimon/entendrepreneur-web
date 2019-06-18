from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_sslify import SSLify
from flask_bootstrap import Bootstrap
from flask_heroku import Heroku
from werkzeug.contrib.fixers import ProxyFix

app = Flask(__name__)
app.config.from_object(Config)
app.wsgi_app = ProxyFix(app.wsgi_app, num_proxies=2) # necessitated by Heroku + CloudFlare (I think)
app.url_map.strict_slashes = False
Bootstrap(app)

if Config.SSLIFY_PERMANENT:
  sslify = SSLify(app)

heroku = Heroku(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

from app import routes, models
