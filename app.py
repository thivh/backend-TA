from flask import Flask, request, jsonify
from model_TA_3 import check_similarity

app = Flask(__name__)

@app.route('/')
def home():
    return 'This server is running'

@app.route('/check_sim/', methods=['GET', 'POST'])
def check():
    text = request.json['transkrip']
    kamus = request.json['kamus']
    return jsonify(check_similarity(text, kamus))



if __name__ == '__main__':
    app.debug = True
    app.run()