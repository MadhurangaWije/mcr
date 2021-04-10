import flask
from flask import request, jsonify

from simpletransformers.question_answering import QuestionAnsweringModel

model = QuestionAnsweringModel('xlmroberta','./trained_model/', use_cuda=False, args={'fp16':False})

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/predict', methods=['POST'])
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
def home():
    return "Welcome to Sinhala Machine Comprehension API by Chamumi Abeysinghe"