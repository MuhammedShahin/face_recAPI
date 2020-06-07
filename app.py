from flask import Flask, request, jsonify, render_template
from Model_functions import *
from keras.models import load_model
import tensorflow as tf
from PIL import Image

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
    return jsonify(identify(img, model, graph))


@app.route('/add_api',methods=['POST'])
def add_api():
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)
    payload = request.form.to_dict()
    id = payload['id']
    name = payload['name']
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


