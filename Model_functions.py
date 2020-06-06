import numpy as np
import cv2
from scipy.spatial import distance


# This line to ignore np package FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


image_size = 160
cascade_path = 'modelFiles/haarcascade_frontalface_alt2.xml'
model_path = 'modelFiles/facenet_keras.h5'


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

    (x, y, w, h) = faces[0]
    cropped = img[y - margin // 2:y + h + margin // 2,
              x - margin // 2:x + w + margin // 2, :]
    # aligned = resize(cropped, (image_size, image_size), mode='reflect')
    aligned = cv2.resize(cropped, (image_size, image_size))
    return aligned


def encoding(img, model, graph):
    with graph.as_default():
        pred = model.predict(np.expand_dims(img, axis=0))
        emb = l2_normalize(pred)
        return emb


def get_diff(img1, model, graph):
    img2 = cv2.imread('shahin1.jpg', cv2.IMREAD_COLOR)

    image1 = prewhiten(face_align(img1))
    image2 = prewhiten(face_align(img2))

    embs1 = encoding(image1, model, graph)
    embs2 = encoding(image2, model, graph)
    return distance.euclidean(embs1, embs2)
