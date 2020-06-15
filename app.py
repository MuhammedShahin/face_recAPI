from flask import Flask, request, jsonify, render_template
from Model_functions import *
from keras.models import load_model
import tensorflow as tf
from PIL import Image
from keras import backend as k
from Facenet_database import *


#Database


# This line to ignore np package FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


global graph
graph = tf.get_default_graph()
model_path = './modelFiles/facenet_keras.h5'
model = load_model(model_path, compile=False)
k.set_learning_phase(0)

app = Flask(__name__)

# students that wnat to take attendance
# global list_of_students
# list_of_students=[]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/TA_login',methods=['POST'])
def TA_login_api():

    data = request.form.to_dict()
    email = data['email']
    password = data['password']
    # id, subjects, state = login_TA(email, password)
    return jsonify(login_TA(email, password))


@app.route('/get_sections', methods=['POST'])
def get_sections():
    data = request.form.to_dict()
    TAID = data['TA_id']
    subID = data['sub_id']
    # lst_sections, year = get_sections_subject(TAID, subID)
    return jsonify(get_sections_subject(TAID, subID))


# @app.route('/prepare_students_ofsection', methods=['POST'])
# def prepare_students_ofsection():
#     data = request.form.to_dict()
#     section_number = data['section_number']
#     year = data['year']
#     list_of_students = get_students_outof_section(section_number , year)
#     # print(list_of_students)
#     # return jsonify(list_of_students)
#     return "Done prepare students for attendance!"


@app.route('/take_attendance', methods=['POST'])
def record_attendance():
    file = request.files['Image']
    # Read the image via file.stream
    img = Image.open(file.stream)
    img = np.array(img)

    data = request.form.to_dict()
    section_number = data['section_number']
    year = data['year']
    week = data['week']
    subject = data['subject']
    list_of_students = get_students_outof_section(section_number, year)
    matched_name, id, min_dist = identify_dataset(img, model, graph, list_of_students)
    if matched_name != None:
        insert_attendance(id, subject, week)
        return jsonify(matched_name, subject, "Recorded")
    else:
        return jsonify(matched_name, subject, "Not_Recorded as their is no match found!!")


@app.route('/check_diff', methods=['POST'])
def check_diff():
    file1 = request.files['Image1']
    file2 = request.files['Image2']
    # Read the image via file.stream
    img1 = Image.open(file1.stream)
    img1 = np.array(img1)

    img2 = Image.open(file2.stream)
    img2 = np.array(img2)

    same = get_diff(img1, img2, model, graph)
    if same:
        return 'Same Person!'
    return 'different persons!'




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
    data = request.form.to_dict()
    id = data['ID']
    name = data['Name']
    section = data['section']
    email = data['email']
    year = data['year']
    img = np.array(img)

    # Encode image
    image1 = prewhiten(face_align(img))
    encod = encoding(image1, model, graph)
    encod = np.asarray(encod)
    print(type(encod))

    insert_student(id, name, email, encod, section, year)
    # dic = {id: [name, encod]}
    # pickle.dump(dic, open("save.p", "wb"))
    # dic = pickle.load(open("save.p", "rb"))
    # if id not in dic:
    #     dic[id] = [name, encod]
    #     pickle.dump(dic, open("save.p", "wb"))
    #     print(dic)
    #     return 'Added successfully'

    return 'Already exist'




