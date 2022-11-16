from yolov7_mod.frame_detection import *
from yolov7_mod.frame_pose_stim import *
from yolov7_segmentation.frame_segment import *
import os

def detect(image_path):
    old_name = r"models_mod"
    new_name = r"models"
    # Renaming the file
    os.rename(old_name, new_name)
    resp = frame_detect(image_path)
    old_name = r"models"
    new_name = r"models_mod"
    # Renaming the file
    os.rename(old_name, new_name)
    return resp
    
def keyp(image_path):
    old_name = r"models_mod"
    new_name = r"models"
    # Renaming the file
    os.rename(old_name, new_name)
    resp = pose_stimation(image_path)
    old_name = r"models"
    new_name = r"models_mod"
    # Renaming the file
    os.rename(old_name, new_name)
    return resp
    
def segment(image_path):
    old_name = r"models_seg"
    new_name = r"models"
    # Renaming the file
    os.rename(old_name, new_name)
    resp = fsegment(image_path)
    old_name = r"models"
    new_name = r"models_seg"
    # Renaming the file
    os.rename(old_name, new_name)
    return resp
 
path = "../datos/orion.jpg"

#detect(path)
#keyp(path)

segment(path)
