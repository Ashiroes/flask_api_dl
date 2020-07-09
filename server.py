from flask import Flask, request, jsonify, send_file

from flask_api_dl.generation import *

app = Flask(__name__)

# GLOBAL
categories = ['Rock', '80s']
categories_with_models = {'Rock': 'test', '80s': 'test2'}


# DEF
def getList(dict: dict) -> list:
    return list(dict.keys())


# FLASK API

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Deep Learning - music </h1>
<p>A prototype API for generate music with Deep Learning.</p>'''


@app.route('/api/v1/categories', methods=['GET'])
def api_all():
    print(categories_with_models.keys())
    return jsonify(getList(categories_with_models))


@app.route('/api/v1/generate_model', methods=['POST'])
def generate_model():
    data = request.json
    category = data['category']

    if category is None:
        resp = jsonify(error="please enter a category")
        return resp

    resp = jsonify(success=True)
    return resp


@app.route('/api/v1/song', methods=['POST'])
def generate_model():
    data = request.json
    category = data['category']
    if category is None:
        resp = jsonify(error="please enter a category")
        return resp

    filename = generate_song(category, categories_with_models[category])

    return send_file(filename_or_fp=filename, mimetype="audio/midi", as_attachment=True)


app.run(debug=True, port=5000)
