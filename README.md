## Entendrepreneur

The Entendrepreneur (entendre/entrepreneur) pun generator creates humorous portmanteaus and rhymes from provided input words.

Example generated portmanteaus:

* **Inputs:** drunk angry → **Output:** beeritable (beer/irritable)
* **Inputs:** cute dog → **Output:** labradorable (labrador/adorable)
* **Inputs:** literary cage → **Output:** shackademic (shack/academic)

For additional details see the associated NeurIPS 2018 workshop paper: [Entendrepreneur: Generating Humorous Portmanteaus using Word-Embeddings](https://nips2018creativity.github.io/doc/entendrepreneur.pdf)

## Requirements

Code was built on Mac OS 10.12.6 using Python 2.7.10

## Usage

Citable paper soon to be released on ArXiv.

## Local App Setup

Install the required Python packages
```
> pip install -r requirements.txt
```

Download 'WordNet' and 'CMU Pronouncing Dictionary' via the nltk downloader (see [here](http://www.nltk.org/data.html)):
```
> python -m nltk.downloader wordnet
> python -m nltk.downloader cmudict
```

Install Homebrew
```
> /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Install Postgres
```
> brew services start postgresql
```

Setup Postgres db (see [here](https://www.codementor.io/engineerapart/getting-started-with-postgresql-on-mac-osx-are8jcopb))
```
> psql postgres
> CREATE ROLE [USERNAME] WITH LOGIN PASSWORD '[PASSWORD]';
> CREATE DATABASE entendrepreneur_db;
```

Export the credentials to `.bash_profile`
```
> echo "export PUN_DATABASE_URL=postgresql://[USERNAME]:[PASSWORD]@localhost/entendrepreneur_db" >> ~/.bash_profile
> python -c "import os; print os.urandom(24).encode('hex')" | read var ; echo "export PUN_SECRET_KEY=$var" >> ~/.bash_profile
> source ~/.bash_profile
```

Initialize the database and populate tables (takes ~10min)
```
> flask db init
> flask db migrate -m 'create tables'
> flask db upgrade
> python manage.py populate_tables
```

Run the app, and go to the url shown to view in browser
```
> flask run
```

## Updating the Database Schema

Note: The following scripts should all be run from within the repo's top-level directory

To delete all rows from a table while keeping the schema intact:
```
> from app import db
> from app.models import TABLENAME
> TABLENAME.query.delete()
> db.session.commit()
```

To drop a table entirely:
```
> from app import db
> from app.models import TABLENAME
> TABLENAME.__table__.drop(db.engine)
> db.session.commit()
```

To create a new (empty) table:
```
> from app import db
> from app.models import TABLENAME
> db.create_all()
```

To copy the Postgres database from local hosting to Heroku hosting:

(1) Dump the data to a file:
```
> PGPASSWORD=punsaregreat pg_dump -Fc --no-acl --no-owner -h localhost -U pun_user entendrepreneur_db > data/db_dumps/entendrepreneur_db_dump_compressed.dump
```
(2) Upload the `entendrepreneur_db_dump_compressed.dump` file to an S3 bucket

(3) Overwrite the current Heroku databse with the content in the S3 file
```
> heroku pg:backups:restore '[S3_URL]' DATABASE_URL --confirm entendrepreneur-app
```
(4) Manually verify the content of the Heroku database
```
> heroku pg:psql
> \dt
```
