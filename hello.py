from flask import Flask, jsonify, request
from worker import Worker
from model import MnistModel
app = Flask(__name__)


mnist = MnistModel()

@app.route("/")
def hello():
    return "Hello, Nghi"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    params = request.json
    worker = Worker()
    strArr = params['array']
    if len(strArr) > 784:
        return jsonify({'code': 500,
                        'message': 'invalid input'})
    bitmap = worker.toBitmapMatrix(strArr)
    answer = mnist.predict(bitmap)['predict_num']
    return jsonify({'code': 200,
                    'message': 'OK',
                    'data': answer})

if __name__ == "__main__":
    app.run()