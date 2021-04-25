import flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin

from simpletransformers.question_answering import QuestionAnsweringModel

model = QuestionAnsweringModel('xlmroberta','./trained_model/', use_cuda=False, args={'fp16':False})

app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():

    request_data = request.json
    print(request_data)
    
    predictions, raw_outputs = model.predict(
        [
            {
                "context": request_data.get('context', ''),
                "qas": [
                    {
                        "question": request_data.get('question', ''),
                        "id": "0",
                    }
                ],
            }
        ]
    )
    answer = predictions[0]['answer'][0]

    return jsonify(answer=str(answer))

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return "Welcome to Sinhala Machine Comprehension API by Chamumi Abeysinghe"


# app.run()
if __name__ == "__main__":
    app.run(debug=True)