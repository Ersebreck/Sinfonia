#!/usr/bin/python

import cv2
import easyocr
import copy
import matplotlib.pyplot as plt
import time

def filter(result, umbral=0.3):
    resp = []
    for res in result:
        if res[-1] >= umbral:
            resp.append(res)
    return resp


def text_extraction(img, rot=[90, 180, 270],gpu=False):
    reader = easyocr.Reader(["en", "es"], gpu=gpu)
    result = reader.readtext(img, paragraph=False, rotation_info=rot, text_threshold=0.7)
    return filter(result)


def rotations(img):
    return [img, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(img, cv2.ROTATE_180),
            cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)]


def r_pred(images, predictions):
    n, m = 2, 2
    angles = [0, 90, 180, 270]
    for i in range(len(images)):
        plt.subplot(n, m, i + 1)
        plt.imshow(visual_pred(images[i], predictions[i]))
        plt.axis("off")
        # plt.title(angles[f"Prediction\nRot:{angles[i]}"])
    plt.tight_layout()


def visual_pred(img, result):
    for res in result:
        try:
            pt0 = res[0][0]
            pt1 = res[0][1]
            pt2 = res[0][2]
            cv2.rectangle(img, pt0, (pt1[0], pt1[1] - 23), (255, 0, 0), -1)
            cv2.putText(img, res[1] + F" P:{round(res[-1], 2)}", (pt0[0], pt0[1] - 3), 2, 0.8, (255, 255, 255), 1)
            cv2.rectangle(img, pt0, pt2, (255, 0, 0), 2)
        except:
            pass
    return img


def final_pred(original, pred):
    n, m = 1, 2
    plt.subplot(n, m, 1)
    plt.imshow(original)
    plt.axis("off")
    plt.title("Original")
    plt.subplot(n, m, 2)
    plt.imshow(pred)
    plt.axis("off")
    plt.title("Prediction")


def run(name, gpu=False, rot=[90,180,270]):
    image = plt.imread(name)
    original = copy.deepcopy(image)
    text = text_extraction(image, gpu=gpu, rot=rot)
    prediction = visual_pred(image, text)
    #final_pred(original, prediction)
    #plt.show()


def camara():
    cap = cv2.VideoCapture(0)
    acceso, frame = cap.read()
    if acceso:
        print(text_extraction(frame))
    else:
        print("Error al acceder a la cámara")

    cap.release()


def pruebas_rot(name):
    r = rotations(plt.imread(name))
    p = []
    for i in r:
        p.append(text_extraction(i))
    r_pred(r, p)

t0 = time.time()

run("libros.jpg", gpu=False)

t1 = time.time()

total = t1-t0
print("F",total)

t0 = time.time()

run("libros.jpg", gpu=True)

t1 = time.time()

total = t1-t0
print("T",total)
print("Terminado")


