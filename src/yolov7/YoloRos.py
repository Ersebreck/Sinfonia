#from yolov7.yolov7_mod.frame_detection import *
#from yolov7.yolov7_mod.frame_pose_stim import *
from yolov7.yolov7_segmentation.frame_segment import *
import os

"""
def detect(image_path):
    cambio_carpeta("models_mod","models")
    resp = frame_detect(image_path)
    cambio_carpeta("models","models_mod")
    return resp
    
def keyp(image_path):
    cambio_carpeta("models_mod","models")
    resp = pose_stimation(image_path)
    cambio_carpeta("models","models_mod")
    
    return resp
"""   

def yolo_segment(image_path):
    #cambio_carpeta("models_seg","models")
    resp = fsegment(image_path)
    #cambio_carpeta("models","models_seg")
    return resp
    
def cambio_carpeta(old, new):
    os.rename(old, new)

 