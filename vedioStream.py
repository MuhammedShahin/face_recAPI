import numpy as np
import cv2
from scipy.spatial import distance
from keras.models import load_model
import tensorflow as tf
from keras import backend as k
import matplotlib.pyplot as plt
import dlib
from align import AlignDlib
import pickle
from PIL import Image
import io
import requests


# image_size = 160
# cascade_path = 'modelFiles/haarcascade_frontalface_alt2.xml'
# threshold = 1.12
# global graph
# graph = tf.get_default_graph()
# model_path = './modelFiles/facenet_keras.h5'
# model = load_model(model_path, compile=False)
# k.set_learning_phase(0)



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


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def face_align(img, margin=10):
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(img,
                                     scaleFactor=1.1,
                                     minNeighbors=3)

    if len(faces) == 0:
        return faces, (-1, -1, -1, -1)
    (x, y, w, h) = faces[0]
    locations = (x, y, w, h)
    cropped = img[y - margin // 2:y + h + margin // 2,
              x - margin // 2:x + w + margin // 2, :]
    # aligned = resize(cropped, (image_size, image_size), mode='reflect')
    aligned = cv2.resize(cropped, (image_size, image_size))
    return aligned, locations


def encoding(img, model, graph):
    with graph.as_default():
        pred = model.predict(np.expand_dims(img, axis=0))
        emb = l2_normalize(pred)
        return emb


alignment = AlignDlib('landmarks.dat')


def face_align11(image):
    img = image.astype(np.uint8)
    bb1 = alignment.getLargestFaceBoundingBox(img, skipMulti=True)
    aligned_image = alignment.align(160, img, bb1, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    koko = 'hello'
    return aligned_image, bb1

def identify(img, model, graph):

    data = pickle.load(open('save.p', 'rb'))
    pred = encoding(img, model, graph)
    embs = l2_normalize(pred)
    min_dist = 1000
    matched_name = 'None'
    for val in data.values():
        dist = distance.euclidean(val[1], embs)
        if dist < min_dist and dist < threshold:
            matched_name = val[0]
            min_dist = dist

    return matched_name, min_dist


def imgTofile(img):
    im_pil = Image.fromarray(img)
    b = io.BytesIO()
    im_pil.save(b, 'jpeg')
    im_bytes = b.getvalue()
    return im_bytes


url = 'http://192.168.1.103:5000/take_attendance'


def start():
    # img = cv2.imread('kalb.PNG')
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # img_rgb = prewhiten(img_rgb)
    # plt.imshow(img_rgb)
    # plt.show()
    # face = face_align11(img_rgb)
    # image_face = prewhiten(face)
    # plt.imshow(face)
    # plt.show()
    # plt.imshow(image_face)
    # plt.show()

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        # face, bb = face_align11(img)
        # # if len(face) == 0:
        # #     continue
        # # prewhit_img = prewhiten(face)
        # # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        # if bb is None:
        #     continue
        # x, y, w, h = bb.left()+10, bb.top()+10, bb.width()+10, bb.height()+10
        # prewhit_img = prewhiten(face)
        # name, min_dist = identify(prewhit_img, model, graph)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(img, "Face : " + name, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        # cv2.putText(img, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        img_file = imgTofile(img)
        file = {'Image': img_file}
        data = {'section_number':12, 'week':12, 'subject':'OS', 'year':3}
        r = requests.post(url, files=file, data=data)
        print(r.text)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('hello', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None


start()
