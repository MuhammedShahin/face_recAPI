from flask import Flask, request, jsonify, render_template
from Model_functions import *
from keras.models import load_model
import tensorflow as tf
from PIL import Image
from keras import backend as k
import json


# This line to ignore np package FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


global graph
graph = tf.get_default_graph()
model_path = './modelFiles/facenet_keras.h5'
model = load_model(model_path)
k.set_learning_phase(0)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/try', methods=['POST'])
def try_():
    data = request.get_json(force=True)
    print(data['name'])
    return data['name']


@app.route('/predict_api', methods=['POST'])
def predict_api():

    file = request.files['Image']
    # Read the image via file.stream
    img = Image.open(file.stream)
    img = np.array(img)

    return jsonify(identify(img, model, graph))


@app.route('/add_api',methods=['POST'])
def add_api():
    file = request.files['Image']
    # Read the image via file.stream
    img = Image.open(file.stream)
    payload = request.form.to_dict()
    id = payload['ID']
    name = payload['Name']
    img = np.array(img)

    # Encode image
    image1 = prewhiten(face_align(img))
    encod = encoding(image1, model, graph)

    # dic = {id: [name, encod]}
    # pickle.dump(dic, open("save.p", "wb"))
    dic = pickle.load(open("save.p", "rb"))
    if id not in dic:
        dic[id] = [name, encod]
        pickle.dump(dic, open("save.p", "wb"))
        print(dic)
        return 'Added successfully'

    return 'Already exist'



# if __name__ == "__main__":
#     app.run(debug=True)


