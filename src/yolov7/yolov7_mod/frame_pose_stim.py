import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from yolov7_mod.utils.datasets import letterbox
from yolov7_mod.utils.general import non_max_suppression_kpt
from yolov7_mod.utils.plots import output_to_keypoint, plot_skeleton_kpts


def pose_stimation(image_path, name="exp", proyect="key"):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load('yolov7_mod/yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    _ = model.float().eval()
    
    if torch.cuda.is_available():
        model.half().to(device)
        
    image = cv2.imread(image_path)
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    
    if torch.cuda.is_available():
        image = image.half().to(device)   
    output, _ = model(image)
    
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    
    cv2.imshow(",",nimg)
    cv2.waitKey(0)
    cv2.imwrite("runs/key/exp/uni.jpg", nimg)
    print("\n\nImage saved at runs/key/exp/uni.jpg \n\n")
    return nimg, output
