from yolov7_segmentation.segment import predict

def fsegment(image_path):
    return predict.run(source=image_path)