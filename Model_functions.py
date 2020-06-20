import numpy as np
import cv2
from scipy.spatial import distance
import pickle


# This line to ignore np package FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


image_size = 160
cascade_path = 'modelFiles/haarcascade_frontalface_alt2.xml'
model_path = 'modelFiles/facenet_keras.h5'
threshold = 1.12


class FaceNet:
    @staticmethod
    def prewhiten(x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    @staticmethod
    def l2_normalize(x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

    @staticmethod
    def face_align(img, margin=10):
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=3)
        if len(faces) == 0:
            return []
        (x, y, w, h) = faces[0]
        cropped = img[y - margin // 2:y + h + margin // 2,
                  x - margin // 2:x + w + margin // 2, :]
        # aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned = cv2.resize(cropped, (image_size, image_size))
        return aligned

    @staticmethod
    def encoding(img, model, graph):
        with graph.as_default():
            pred = model.predict(np.expand_dims(img, axis=0))
            # emb = l2_normalize(pred)
            return pred


def get_diff(img1, img2, model, graph):

    face1 = FaceNet.face_align(img1)
    face2 = FaceNet.face_align(img2)
    if face1 == [] or face2 == []:
        return 'No face found in photo'
    image1 = FaceNet.prewhiten(face1)
    image2 = FaceNet.prewhiten(face2)

    pred1 = FaceNet.encoding(image1, model, graph)
    embs1 = FaceNet.l2_normalize(pred1)
    pred2 = FaceNet.encoding(image2, model, graph)
    embs2 = FaceNet.l2_normalize(pred2)
    dist = distance.euclidean(embs1, embs2)
    if dist < threshold:
        return 'Same Person!'
    return 'different persons!'


def identify(img, model, graph):

    data = pickle.load(open('save.p', 'rb'))
    image = FaceNet.prewhiten(FaceNet.face_align(img))
    pred = FaceNet.encoding(image, model, graph)
    embs = FaceNet.l2_normalize(pred)
    min_dist = 1000
    matched_name = None
    for val in data.values():
        dist = distance.euclidean(val[1], embs)
        if dist < min_dist and dist < threshold:
            matched_name = val[0]
            min_dist = dist

    return matched_name, min_dist


def identify_dataset(img, model, graph, list_of_students):
    face = FaceNet.face_align(img)
    if face == []:
        return 'NO Face Found in photo', None, None
    image = FaceNet.prewhiten(face)
    pred = FaceNet.encoding(image, model, graph)
    embs = FaceNet.l2_normalize(pred)
    min_dist = 1000
    matched_name = None
    id = None
    for val in list_of_students:
        print(val[2])
        dist = distance.euclidean(val[2], embs)
        if dist < min_dist and dist < threshold:
            matched_name = val[1]
            id = val[0]
            min_dist = dist

    return matched_name, id , min_dist


# from Model_functions import *
# from keras.models import load_model
# import tensorflow as tf
# from Facenet_database import *
# global graph
# graph = tf.get_default_graph()
# model_path = './modelFiles/facenet_keras.h5'
# model = load_model(model_path, compile=False)
# def main():
#     list_of_students = get_students_outof_section(3, 2)
#     image1 = cv2.imread('gomaa0.jpg')
#     img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#     matched_name, id , min_dist = identify_dataset(img, model, graph, list_of_students)
#     koko=0
#
#
# main()