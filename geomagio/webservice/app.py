from __future__ import absolute_import, unicode_literals

import os
import flask

from . import database
from . import login
from . import session


def create_app():
    app = flask.Flask(__name__)
    # configure using environment variables
    app.config.update(os.environ)

    # connect modules
    database.init_app(app)
    login.init_app(app)
    session.init_app(app)

    # add default route
    @app.route('/')
    def index():
        return flask.render_template('index.html')

    # add commands
    #@app.cli.command('create_db')
    #def create_db():
    #    # create database tables
    #    database.db.create_all()

    return app