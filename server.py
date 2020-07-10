import base64
import os
from os.path import dirname, realpath, join

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from music21 import instrument
from werkzeug.utils import secure_filename

from generation import generate_song
from style_transfer import style_transfer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './upload/'
ALLOWED_EXTENSIONS = set(['jpg', 'png'])

cors = CORS(app, resources={r"/*": {"origins": "*"}})


# GLOBAL
basedir = os.path.abspath(os.path.dirname(__file__))

categories_with_models = {'Rock': 'test', '80s': 'modeles/80s.hdf5'}
instruments = {'Guitare': instrument.AcousticGuitar(), 'Piano': instrument.Piano(), 'Violon': instrument.Violin()}

# DEF

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getList(dict: dict) -> list:
    return list(dict.keys())

def validate_file(request, name):
    if name not in request.files:
        return False
    file = request.files[name]
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return False
    if file and allowed_file(file.filename):
        return True

# FLASK API

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Deep Learning - music </h1>
<p>A prototype API for generate music with Deep Learning.</p>'''


@app.route('/api/v1/categories', methods=['GET'])
def api_get_all_categories():
    return jsonify(getList(categories_with_models))


@app.route('/api/v1/instruments', methods=['GET'])
def api_get_all_instruments():
    return jsonify(getList(instruments))

@app.route('/api/v1/generate_image', methods=['POST'])
def api_post_create_image():
    if validate_file(request, 'image') and validate_file(request, 'style'):
        image = request.files['image']
        style = request.files['style']
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename)))
        style.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style.filename)))
        filename = style_transfer(
            os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename)),
            os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style.filename)))

        return jsonify(data=filename)

    #        return send_file(filename_or_fp=filename, mimetype="image/jpeg", as_attachment=True)
    return jsonify(error="incompatible files or files not given.")


@app.route('/api/v1/generate_song', methods=['POST'])
def api_post_create_song():
    data = request.json

    category = data.get('category')
    if category is None:
        return jsonify(error="please enter a category")
    if category not in getList(categories_with_models):
        return jsonify(error="incorrect category : " + category + " not in " + str(getList(categories_with_models)))

    instru = data.get('instrument')
    if instru is not None:
        if instru not in getList(instruments):
            return jsonify(error="incorrect instrument : " + instru + " not in " + str(getList(instruments)))
        filename = generate_song(category, categories_with_models[category], instruments[instru])
    else:
        filename = generate_song(category, categories_with_models[category])

    return send_file(filename_or_fp=filename, mimetype="audio/midi", as_attachment=True)

app.run(debug=True, port=5000)
