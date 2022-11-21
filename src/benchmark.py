import os, platform, subprocess, re, sys
import time
import numpy as np
from timeit import default_timer as timer
import depth
import csv
import cv2

from yolov7.YoloRos import yolo_segment, yolo_detect

# obtain absolute path to the directory containing this file
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_NAME = os.path.dirname(DIR_PATH)

print(DIR_NAME)
DEPTH_FOLDER = DIR_NAME+"/src/images/depth"
OUTPUT_FOLDER = DIR_NAME+"/src/output/temp"
IMG_FOLDER = DIR_NAME+"/src/images/img"
TEMP = DIR_NAME + "/src/result_img"
TEST = DIR_NAME + "/src/output/"
TEST_FILENAME = "results.csv"
TESTS_FOLDER = TEST+ TEST_FILENAME


def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""


def test_segment(data, method, path):
    rgb_image, depth_image = data
    # x1,y1,x2,y2,label = object.object_detection(path) TODO
    pred = yolo_segment(path)
    if len(pred[1])>0:
      arr = pred[1][0]
      result = pred[0]
      x1,y1,x2,y2,label = int(arr[0]),int(arr[1]),int(arr[2]),int(arr[3]),int(arr[4])
      tiempo_detect = int(pred[2]) # este es el tiempo que demora la prediccion
      cx = 10
      cy = 10
      total_time = tiempo_detect
      bounding_box = y1,x1,y2,x2
      return img, total_time, result
    else:
        return None, 0,0,0


def test_cpu(data, method,path):
    rgb_image, depth_image = data
    start = timer()
    # x1,y1,x2,y2,label = object.object_detection(path) TODO
    pred = yolo_detect(path)
    if len(pred[1])>0:
      arr = pred[1][0]
      result = pred[0]
      x1,y1,x2,y2,label = int(arr[0]),int(arr[1]),int(arr[2]),int(arr[3]),int(arr[4])
      tiempo_detect = int(pred[2])
      end = timer()
    else:
        return None, 0,0,0

    if len(pred) >0:
        cx = 10
        cy = 10
        total_time = end-start
        bounding_box = y1,x1,y2,x2
        segmentating_time=0
        if method == "iterative":
            img, segmentating_time = depth.segmentate_iterative(depth_image, bounding_box)
        else:
            img,segmentating_time = depth.object_border(depth_image, bounding_box)
        total_time+=segmentating_time
        return img, total_time, segmentating_time,result
    return None, 0,0,0

def main():
    proccesor = get_processor_name().strip()
    os_name = platform.system()
    os_version = platform.platform()
    # method = "iterative"
    method = "quick_iterative" 
    N = 10
    print("Processor: ", proccesor)
    print("OS: ", os_name)
    print("OS version: ", os_version)

    header = ["processor", "os", "os_version", "file", "time_avg_complete" , "time_avg_segmentating", "n", "method"]
    # save results to csv file in the folder tests
    if not os.path.exists(TESTS_FOLDER):
        os.makedirs(TESTS_FOLDER)
    if not os.path.exists(TESTS_FOLDER + "/" + TEST_FILENAME):
        f = open(TESTS_FOLDER+"/"+TEST_FILENAME, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(header)
    else:
        f = open(TESTS_FOLDER+"/"+TEST_FILENAME, "a", newline="")
        writer = csv.writer(f)

    # get all csv files in the folder DEPTH_FOLDER
    files = os.listdir(DEPTH_FOLDER)
    csv_files = [file for file in files if file.endswith(".csv")]
    for file in csv_files:
        pic_name = file[:-4]
        total_time_final = 0
        segmentating_time_final = 0
        data = cv2.imread(IMG_FOLDER+"/"+pic_name+".jpg"), np.loadtxt(DEPTH_FOLDER+"/"+file, delimiter=",")
        for _ in range(N):
            #load the image in rgb and the depth image
            if method == "iterative":
                img, total_time, segmentating_time,result = test_cpu(data, method,IMG_FOLDER+"/"+pic_name+".jpg")
            elif method == "quick_iterative":
                img, total_time, segmentating_time,result = test_cpu(data, method, IMG_FOLDER+"/"+pic_name+".jpg")
            else:
                print("Invalid method")
                return
            total_time_final += total_time
            segmentating_time_final += segmentating_time
            break
        if img is None:
            continue


        avg = total_time_final / N
        avg_segmentating = segmentating_time_final / N
        print("Time for file: ", file, " is: ", avg)
        test = [proccesor, os_name, os_version, file, avg, avg_segmentating, N, method]
        writer.writerow(test)
        # save image to the folder OUTPUT_FOLDER
        
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        if method == "quick_iterative":
            border = img
            img = result
            cx = 10
            for border_point in border:
                i, j = border_point
                try:
                    img[i+cx][j-cx]=(0,255,0)
                except:
                    continue
            cv2.imwrite(OUTPUT_FOLDER+"/"+method+pic_name+".jpg", img)
        # print(img)
        else:
            np.savetxt(OUTPUT_FOLDER+"/"+method+file, img, delimiter=",", fmt="%d")
    f.close()
    # transform_csv_to_img()

def transform_csv_to_img():
    files = os.listdir(OUTPUT_FOLDER)
    csv_files = [file for file in files if file.endswith(".csv")]
    for file in csv_files:
        data = np.loadtxt(OUTPUT_FOLDER+"/"+file, delimiter=",")
        depth.save_img(data, TEMP+"/"+file[:-4]+".png")

if __name__ == '__main__':
    main()
