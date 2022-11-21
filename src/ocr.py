import cv2 
import pytesseract
import rospkg
from PIL import Image
import os

file_name = 'cedulafm.jpg'
PATH_OCR ='../resources/data/ocr/'

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


print(pytesseract.image_to_boxes(PATH_OCR+file_name))

img = cv2.imread(PATH_OCR+file_name)
img = cv2.resize(img, dsize = None,fx=0.8, fy=0.8)
gray = get_grayscale(img)
thresh = thresholding(gray)

# Adding custom optionss
custom_config = r'-l spa --psm 6'

print(pytesseract.image_to_string(thresh, config=custom_config))

h, w, c = img.shape
boxes = pytesseract.image_to_boxes(thresh) 
for b in boxes.splitlines():
    b = b.split(' ')
    thresh = cv2.rectangle(thresh, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

cv2.imshow('img', thresh)
cv2.waitKey(0)

