import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
import json
import datetime

from flask import Flask, Response, request, session, render_template
from wtforms import Form, TextField, SubmitField

from engine import Engine
from movies import MovieDict

# def create_app():
# global engine
engine = Engine()
movie_dict = MovieDict(engine.spark)
app = Flask(__name__)
app.secret_key = os.urandom(16).hex()
# return app
# app = create_app()

class SearchForm(Form):
    autocomp = TextField('title:', id='movie_autocomplete')
    submit = SubmitField('rate')

@app.route('/_autocomplete', methods=['GET'])
def autocomplete():
    return Response(
        json.dumps(list(movie_dict.options.keys())),
        mimetype='application/json'
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    session['idratings'] = {}
    session['userratings'] = {}
    return render_template(
        'index.html',
        form=SearchForm(request.form)
    )

@app.route('/submitRating', methods=['POST'])
def submitRating():
    movie = str(request.form['autocomp'])
    session['movie'] = movie
    if movie not in movie_dict.options:
        return render_template(
            'movies.html',
            form=SearchForm(request.form),
            userratings=session.get('userratings')
        )
    return render_template('rate.html', movie=movie)

@app.route('/movies', methods=["POST"])
def movies():
    movie = session['movie']
    rating = int(request.form['rating'])
    user_ratings = session.get('userratings')
    id_ratings = session.get('idratings')
    user_ratings[movie] = rating
    movie_id = int(movie_dict.options[movie])
    id_ratings[movie_id] = rating
    session['movie'] = None
    return render_template(
        'movies.html',
        form=SearchForm(request.form),
        userratings=user_ratings
    )

@app.route('/remove', methods=["POST"])
def remove():
    #remove from user ratings
    remove_movie = request.form['remove_movie']
    user_ratings = session.get('userratings')
    del user_ratings[remove_movie]

    #remove from id ratings
    remove_id = str(movie_dict.options[remove_movie])
    id_ratings = session.get('idratings')
    del id_ratings[remove_id]

    # should this be happening in /movies?
    session['idratings'] = id_ratings
    session['userratings'] = user_ratings

    return render_template(
        'movies.html',
        form=SearchForm(request.form), userratings=user_ratings
    )


@app.route('/makeRecommendations', methods=["POST"])
def makeRecommendations():
    id_ratings = session.get('idratings')
    user_ratings = session.get('userratings')
    recommended_movie_ids = engine.make_recommendations(id_ratings)
    if not recommended_movie_ids:
        recommended_movies = ['no recommendations available']
    else:
        recommended_movies = movie_dict.get_titles(recommended_movie_ids)
    return render_template('results.html', movies=recommended_movies)

if __name__ == '__main__':
    app.run(
        '0.0.0.0',
        debug=True,
        threaded=True
    )
