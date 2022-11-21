import os


def yolo_detect(image_path):
    from yolov7.yolov7_mod.frame_detection import frame_detect
    cambio_carpeta("models_mod","models")
    resp = frame_detect(image_path)
    cambio_carpeta("models","models_mod")
    return resp
    
def yolo_keyp(image_path):
    #from yolov7.yolov7_mod.frame_pose_stim import *
    cambio_carpeta("yolov7/models_mod","yolov7/models")
    resp = pose_stimation(image_path)
    cambio_carpeta("yolov7/models","yolov7/models_mod")
    
    return resp
  

def yolo_segment(image_path):
    from yolov7.yolov7_segmentation.frame_segment import fsegment
    cambio_carpeta("models_seg","models")
    resp = fsegment(image_path)
    cambio_carpeta("models","models_seg")
    return resp
    
def cambio_carpeta(old, new):
    os.rename(old, new)

 