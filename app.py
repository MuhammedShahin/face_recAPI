import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from Model_functions import get_diff
from keras.models import load_model
import tensorflow as tf

# This line to ignore np package FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


global graph
graph = tf.get_default_graph()
model_path = './modelFiles/facenet_keras.h5'
model = load_model(model_path)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():

    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # print(img.shape)
    #
    # response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
    #             }
    # # encode response using jsonpickle
    # response_pickled = jsonpickle.encode(response)

    # return Response(response=response_pickled, status=200, mimetype="application/json")
    # return jsonify(get_diff(img, model, graph))
    return jsonify(get_diff(img, model, graph))


# if __name__ == "__main__":
#     app.run(debug=True)


