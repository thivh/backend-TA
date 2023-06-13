from flask import Flask, request, jsonify
from model_TA import check_similarity

app = Flask(__name__)

@app.route('/')
def home():
    return 'Salam tawa salam bahagia :)'

@app.route('/check_sim/', methods=['GET', 'POST'])
def check():
    text = request.json['transkrip']
    kamus = request.json['kamus']
    return jsonify(check_similarity(text, kamus))



if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug = True)