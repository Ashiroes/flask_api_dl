import flask
from flask import Flask, request, jsonify

app = Flask(__name__)

# GLOBAL
categories = ['Rock', '80s']
categories_with_models = { 'Rock': 'test', '80s': 'test2' }

#DEF
def getList(dict: dict) -> list:
    return list(dict.keys())


#FLASK API

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
    print(category)
    # if category == None:
    #     #todo
    resp = jsonify(success=True)
    return resp


app.run(debug=True, port=5000)
