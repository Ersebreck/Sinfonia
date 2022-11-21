import numpy as np
import time
import cv2
import sys
import threading
import os

sys.setrecursionlimit(76800)

# obtain absolute path to the directory containing this file
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_NAME = os.path.dirname(DIR_PATH)


DEPTH_FOLDER = DIR_NAME+"/resources/data/images/depth"
IMG_FOLDER = DIR_NAME+"/resources/data/images/rgb"
TESTS_FOLDER = DIR_NAME+"/resources/data/images/tests"
TEST_FILENAME = "results.csv"
TEST_PATH = TESTS_FOLDER+"/"+TEST_FILENAME
RESULT_FOLDER = DIR_NAME+"/resources/data/images/results"



def show_img (img):
    # Changes format from float to uint8 as gradient
    uint_img = np.array(img*255).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Image",grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

def save_img (img, name):
    # Changes format from float to uint8 as gradient
    uint_img = np.array(img*255).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(name,grayImage)
    return None

#########################################


centro_mano = (157,157)

fixed_threshold = 30
scalar = 2
corrimiento_i_RGB_depth = -10
corrimiento_j_RGB_depth = 10


############################################# ITERATIVE SEGMENTATION ################################
def segmentate_iterative(image, bounds):
    start = time.time()
    result = np.zeros(image.shape)
    y1,x1,y2,x2 = bounds
    check_gradient = np.full(image.shape,False)
    center = ((y1+y2)//2,(x1+x2)//2)
    # center = centro_mano
    result[center[0],center[1]] = 1
    check_gradient[center[0],center[1]] = True
    stack = [center]
    i_gradient = 0
    j_gradient = 0
    dynamic_i_min = 0
    dynamic_i_max = 0
    dynamic_j_min = 0
    dynamic_j_max = 0

    while len(stack) > 0:
        dynamic_i_min = i_gradient/scalar
        dynamic_i_max = i_gradient*scalar
        dynamic_j_min = j_gradient/scalar
        dynamic_j_max = j_gradient*scalar
        pixel = stack.pop()
        #Check top pixel
        if pixel[0] + 1 < y2:
            # print(pixel[0] + 1,pixel[1])
            # print(x+w,y+h)
            if not check_gradient[pixel[0]+1,pixel[1]]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]])
                #The gradient of the pixel in x direction was already checked
                check_gradient[pixel[0]+1][pixel[1]] = True
                if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    result[pixel[0]+1][pixel[1]] = 1
                    i_gradient = new_gradient
                    stack.append((pixel[0]+1,pixel[1]))
                else:
                    #The pixel is outside the gradient
                    result[pixel[0]+1][pixel[1]] = 0

        #Check bot pixel
        if pixel[0] - 1 > y1:
            if not check_gradient[pixel[0]-1][pixel[1]]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]])
                #The gradient of the pixel in x direction was already checked
                check_gradient[pixel[0]-1][pixel[1]] = True
                if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    result[pixel[0]-1][pixel[1]] = 1
                    i_gradient = new_gradient
                    stack.append((pixel[0]-1,pixel[1]))
                else:
                    #The pixel is outside the gradient
                    result[pixel[0]-1][pixel[1]] = 0

        #Check left pixel
        if pixel[1] - 1 > x1:
            if not check_gradient[pixel[0]][pixel[1]-1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]-1])
                #The gradient of the pixel in x direction was already checked
                check_gradient[pixel[0]][pixel[1]-1] = True
                if dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    result[pixel[0]][pixel[1]-1] = 1
                    j_gradient = new_gradient
                    stack.append((pixel[0],pixel[1]-1))
                else:
                    #The pixel is outside the gradient
                    result[pixel[0]][pixel[1]-1] = 0

        #Check right pixel
        if pixel[1] + 1 < x2:
            if not check_gradient[pixel[0]][pixel[1]+1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]+1])
                #The gradient of the pixel in x direction was already checked
                check_gradient[pixel[0]][pixel[1]+1] = True
                if dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    result[pixel[0]][pixel[1]+1] = 1
                    j_gradient = new_gradient
                    stack.append((pixel[0],pixel[1]+1))
                else:
                    #The pixel is outside the gradient
                    result[pixel[0]][pixel[1]+1] = 0
    return result, time.time() - start



########################################## OBJECT BORDER DETECTION ################################

def object_border(image, bounds):

    start_time = time.time()

    result = []

    y1,x1,y2,x2 = bounds

    check_gradient = np.full(image.shape,False)

    center = ((y1+y2)//2, (x1+x2)//2)
    # center = centro_mano
    # center =center[0]+corrimiento_i_RGB_depth,center[1]+corrimiento_j_RGB_depth

    check_gradient[center[0],center[1]] = True

    gradient = 0

    pixel = center
    bot = [y2,center[1]]

    ############################ FIND UP,DOWN,LEFT,RIGHT BORDERS############ 
    #Iterate to the bot until the gradient is too high
    while pixel[0] + 1 < y2:
        pixel = (pixel[0]+1,pixel[1])
        bot = pixel
        dynamic_i_min = gradient/scalar
        dynamic_i_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]])
        if new_gradient < fixed_threshold or dynamic_i_min < new_gradient < dynamic_i_max:
            gradient = new_gradient
        else:
            bot = [pixel[0]-1,pixel[1]]
            break
    pixel = center
    gradient = 0
    top = [y1,center[1]]

    #Iterate to the top until the gradient is too high
    while pixel[0] - 1 > y1:

        pixel = (pixel[0]-1,pixel[1])
        top = pixel
        dynamic_i_min = gradient/scalar
        dynamic_i_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]])
        if new_gradient < fixed_threshold or dynamic_i_min < new_gradient < dynamic_i_max:
            gradient = new_gradient
        else:
            top = [pixel[0]+1,pixel[1]]
            break

    # print("top done",top)

    pixel = center
    gradient = 0
    left = [center[0],x1]

    #Iterate to the left until the gradient is too high
    while pixel[1] - 1 > x1:
        pixel = (pixel[0],pixel[1]-1)
        left = pixel
        dynamic_j_min = gradient/scalar
        dynamic_j_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]+1])
        if new_gradient < fixed_threshold or dynamic_j_min < new_gradient < dynamic_j_max:
            gradient = new_gradient
        else:
            left = [pixel[0],pixel[1]+1]
            break

    # print("left done",left)

    pixel = center
    gradient = 0
    right = [center[0],x2]

    #Iterate to the right until the gradient is too high
    while pixel[1] + 1 <  x2:
        pixel = (pixel[0],pixel[1]+1)
        right = pixel
        dynamic_j_min = gradient/scalar
        dynamic_j_max = gradient*scalar
        new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]-1])
        if new_gradient < fixed_threshold or dynamic_j_min < new_gradient < dynamic_j_max:
            gradient = new_gradient
        else:
            right = [pixel[0],pixel[1]-1]
            break

    # print("right done",right)

    ###################################### CHECK BORDERS ITERATIVELY ######################
    def check_borders(lock, start):
        new_gradient = 0
        stack = []
        gradient = [0,0]
        stack.append(start)
        while len(stack) > 0:
            pixel = stack.pop()
            #Check left pixel
            if pixel[1] - 1 >= x1 and not check_gradient[pixel[0]][pixel[1]-1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]-1])
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x direction was already checked
                lock.acquire()
                check_gradient[pixel[0]][pixel[1]-1] = True
                lock.release()
                if dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[1] = new_gradient
                    # check the left, top and bottom of the pixel to know if it is a border pixel
                    if pixel[1] - 2 >= x1:
                        left_gradient = abs(image[pixel[0]][pixel[1]-1] - image[pixel[0]][pixel[1]-2])
                    else:
                        left_gradient = fixed_threshold+1
                    if pixel[0] - 1 >= y1:
                        top_gradient = abs(image[pixel[0]][pixel[1]-1] - image[pixel[0]-1][pixel[1]-1])
                    else:
                        top_gradient = fixed_threshold+1
                    if pixel[0] + 1 < y2:
                        bot_gradient = abs(image[pixel[0]][pixel[1]-1] - image[pixel[0]+1][pixel[1]-1])
                    else:
                        bot_gradient = fixed_threshold+1

                    if left_gradient > fixed_threshold or top_gradient > fixed_threshold or bot_gradient > fixed_threshold:
                        # lock.acquire()
                        result.append((pixel[0],pixel[1]-1))
                        # result[pixel[0]][pixel[1]-1] = 1
                        # lock.release()
                        stack.append((pixel[0],pixel[1]-1))


            #Check top pixel
            if pixel[0] - 1 >= y1 and not check_gradient[pixel[0]-1][pixel[1]]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                #The gradient of the pixel in y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]-1][pixel[1]] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    # check the top, left, right of the pixel to know if it is a border pixel
                    if pixel[0] - 2 >= y1:
                        top_gradient = abs(image[pixel[0]-1][pixel[1]] - image[pixel[0]-2][pixel[1]])
                    else:
                        top_gradient = fixed_threshold+1
                    if pixel[1] - 1 >= x1:
                        left_gradient = abs(image[pixel[0]-1][pixel[1]] - image[pixel[0]-1][pixel[1]-1])
                    else:
                        left_gradient = fixed_threshold+1
                    if pixel[1] + 1 <  x2:
                        right_gradient = abs(image[pixel[0]-1][pixel[1]] - image[pixel[0]-1][pixel[1]+1])
                    else:
                        right_gradient = fixed_threshold+1
                    if top_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold:
                        # lock.acquire()
                        result.append((pixel[0]-1,pixel[1]))
                        # result[pixel[0]-1][pixel[1]] = 1
                        # lock.release()
                        stack.append((pixel[0]-1,pixel[1]))

            #Check right pixel
            if pixel[1] + 1 < x2 and not check_gradient[pixel[0]][pixel[1]+1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]][pixel[1]+1])
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x direction was already checked
                lock.acquire()
                check_gradient[pixel[0]][pixel[1]+1] = True
                lock.release()
                if dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[1] = new_gradient
                    # check the right, top, bot pixels of the pixel to know if it is a border pixel
                    if pixel[1] + 2 <  x2:
                        right_gradient = abs(image[pixel[0]][pixel[1]+1] - image[pixel[0]][pixel[1]+2])
                    else:
                        right_gradient = fixed_threshold+1
                    if pixel[0] - 1 >= y1:
                        top_gradient = abs(image[pixel[0]][pixel[1]+1] - image[pixel[0]-1][pixel[1]+1])
                    else:
                        top_gradient = fixed_threshold+1
                    if pixel[0] + 1 < y2:
                        bot_gradient = abs(image[pixel[0]][pixel[1]+1] - image[pixel[0]+1][pixel[1]+1])
                    else:
                        bot_gradient = fixed_threshold+1

                    if right_gradient > fixed_threshold or top_gradient > fixed_threshold or bot_gradient > fixed_threshold:
                        # lock.acquire()
                        result.append((pixel[0],pixel[1]+1))
                        # result[pixel[0]][pixel[1]+1] = 1
                        # lock.release()
                        stack.append((pixel[0],pixel[1]+1))

            #Check bottom pixel
            if pixel[0] + 1 < y2 and not check_gradient[pixel[0]+1][pixel[1]]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                #The gradient of the pixel in y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]+1][pixel[1]] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    # check the bottom, left, right pixels of the pixel to know if it is a border pixel
                    if pixel[0] + 2 < y2:
                        bot_gradient = abs(image[pixel[0]+1][pixel[1]] - image[pixel[0]+2][pixel[1]])
                    else:
                        bot_gradient = fixed_threshold+1
                    if pixel[1] - 1 >= x1:
                        left_gradient = abs(image[pixel[0]+1][pixel[1]] - image[pixel[0]+1][pixel[1]-1])
                    else:
                        left_gradient = fixed_threshold+1
                    if pixel[1] + 1 <  x2:
                        right_gradient = abs(image[pixel[0]+1][pixel[1]] - image[pixel[0]+1][pixel[1]+1])
                    else:
                        right_gradient = fixed_threshold+1

                    if bot_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold:
                        # lock.acquire()
                        result.append((pixel[0]+1,pixel[1]))
                        # result[pixel[0]+1][pixel[1]] = 1
                        # lock.release()
                        stack.append((pixel[0]+1,pixel[1]))

            # check top left corner
            if pixel[0] - 1 >= y1 and pixel[1] - 1 >= x1 and not check_gradient[pixel[0]-1][pixel[1]-1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]-1])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x and y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]-1][pixel[1]-1] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    gradient[1] = new_gradient
                    # check the top, left, right pixels of the pixel to know if it is a border pixel

                    if pixel[0] - 2 >= y1:
                        top_gradient = abs(image[pixel[0]-1][pixel[1]-1] - image[pixel[0]-2][pixel[1]-1])
                    else:
                        top_gradient = fixed_threshold+1
                    if pixel[1] - 2 >= x1:
                        left_gradient = abs(image[pixel[0]-1][pixel[1]-1] - image[pixel[0]-1][pixel[1]-2])
                    else:
                        left_gradient = fixed_threshold+1

                    right_gradient = abs(image[pixel[0]-1][pixel[1]-1] - image[pixel[0]-1][pixel[1]])

                    bot_gradient = abs(image[pixel[0]-1][pixel[1]-1] - image[pixel[0]][pixel[1]-1])

                    if top_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold or bot_gradient > fixed_threshold:
                        # lock.acquire()
                        result.append((pixel[0]-1,pixel[1]-1))
                        # result[pixel[0]-1][pixel[1]-1] = 1
                        # lock.release()
                        stack.append((pixel[0]-1,pixel[1]-1))

            # check top right corner
            if pixel[0] - 1 >= y1 and pixel[1] + 1 < x2 and not check_gradient[pixel[0]-1][pixel[1]+1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]-1][pixel[1]+1])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x and y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]-1][pixel[1]+1] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    gradient[1] = new_gradient
                    # check the top, left, right of the pixel to know if it is a border pixel
                    if pixel[0] - 2 >= y1:
                        top_gradient = abs(image[pixel[0]-1][pixel[1]+1] - image[pixel[0]-2][pixel[1]+1])
                    else:
                        top_gradient = fixed_threshold+1

                    left_gradient = abs(image[pixel[0]-1][pixel[1]+1] - image[pixel[0]-1][pixel[1]])

                    if pixel[1] + 2 <  x2:
                        right_gradient = abs(image[pixel[0]-1][pixel[1]+1] - image[pixel[0]-1][pixel[1]+2])
                    else:
                        right_gradient = fixed_threshold+1

                    bot_gradient = abs(image[pixel[0]-1][pixel[1]+1] - image[pixel[0]][pixel[1]+1])

                    if top_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold or bot_gradient > fixed_threshold:
                        # lock.acquire()
                        result.append((pixel[0]-1,pixel[1]+1))
                        # result[pixel[0]-1][pixel[1]+1] = 1
                        # lock.release()
                        stack.append((pixel[0]-1,pixel[1]+1))

            # check bottom left corner
            if pixel[0] + 1 < y2 and pixel[1] - 1 >= x1 and not check_gradient[pixel[0]+1][pixel[1]-1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]-1])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x and y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]+1][pixel[1]-1] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    gradient[1] = new_gradient
                    # check the bot, left, right of the pixel to know if it is a border pixel
                    bot_gradient = abs(image[pixel[0]+1][pixel[1]-1] - image[pixel[0]][pixel[1]-1])

                    if pixel[1] - 2 >= x1:
                        left_gradient = abs(image[pixel[0]+1][pixel[1]-1] - image[pixel[0]+1][pixel[1]-2])
                    else:
                        left_gradient = fixed_threshold+1

                    right_gradient = abs(image[pixel[0]+1][pixel[1]-1] - image[pixel[0]+1][pixel[1]])

                    top_gradient = abs(image[pixel[0]+1][pixel[1]-1] - image[pixel[0]][pixel[1]-1])

                    if bot_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold or top_gradient > fixed_threshold:
                        # lock.acquire()
                        result.append((pixel[0]+1,pixel[1]-1))
                        # result[pixel[0]+1][pixel[1]-1] = 1
                        # lock.release()
                        stack.append((pixel[0]+1,pixel[1]-1))
            # check bottom right corner
            if pixel[0] + 1 < y2 and pixel[1] + 1 < x2 and not check_gradient[pixel[0]+1][pixel[1]+1]:
                #The pixel is inside the bounds and has not been checked
                new_gradient = abs(image[pixel[0]][pixel[1]] - image[pixel[0]+1][pixel[1]+1])
                dynamic_i_min = gradient[0]/scalar
                dynamic_i_max = gradient[0]*scalar
                dynamic_j_min = gradient[1]/scalar
                dynamic_j_max = gradient[1]*scalar
                #The gradient of the pixel in x and y direction was already checked
                lock.acquire()
                check_gradient[pixel[0]+1][pixel[1]+1] = True
                lock.release()
                if dynamic_i_min < new_gradient < dynamic_i_max or dynamic_j_min < new_gradient < dynamic_j_max or new_gradient < fixed_threshold:
                    #The pixel is inside the gradient and respects or is below the fixed threshold
                    gradient[0] = new_gradient
                    gradient[1] = new_gradient
                    # check the bot, left, right of the pixel to know if it is a border pixel
                    if pixel[0] + 2 < y2:
                        bot_gradient = abs(image[pixel[0]+1][pixel[1]+1] - image[pixel[0]+2][pixel[1]+1])
                    else:
                        bot_gradient = fixed_threshold+1

                    left_gradient = abs(image[pixel[0]+1][pixel[1]+1] - image[pixel[0]+1][pixel[1]])

                    if pixel[1] + 2 < x2:
                        right_gradient = abs(image[pixel[0]+1][pixel[1]+1] - image[pixel[0]+1][pixel[1]+2])
                    else:
                        right_gradient = fixed_threshold+1

                        top_gradient = abs(image[pixel[0]+1][pixel[1]+1] - image[pixel[0]][pixel[1]+1])

                    if bot_gradient > fixed_threshold or left_gradient > fixed_threshold or right_gradient > fixed_threshold or top_gradient > fixed_threshold:
                        # lock.acquire()
                        result.append((pixel[0]+1,pixel[1]+1))
                        # result[pixel[0]+1][pixel[1]+1] = 1
                        # lock.release()
                        stack.append((pixel[0]+1,pixel[1]+1))

    lock = threading.Lock()

    #create threads
    thread1 = threading.Thread(target=check_borders, args=(lock, top))
    thread2 = threading.Thread(target=check_borders, args=(lock, bot))
    thread3 = threading.Thread(target=check_borders, args=(lock, left))
    thread4 = threading.Thread(target=check_borders, args=(lock, right))

    #start threads
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    #wait for threads to finish
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    return result, time.time() - start_time



if __name__ == "__main__":
    # Import Depth Matrix
    file_name = input("Enter the file name: ")

    data= np.loadtxt(DEPTH_FOLDER+ "/" + file_name + ".csv", delimiter=",", skiprows=0, dtype=np.float32)
    img = np.zeros(data.shape)
    # print(data.shape)
    t2 =time.time()
    border_points, el = object_border(data, (93,100,208,239))
    t3 = time.time()

    img1 = segmentate_iterative(data, (0,0,240,320))
    t4 = time.time()
    print("Segmentation iterative time: ", t4-t3)
    print("Segmentation border time: ", el)

    rgb_img = cv2.imread(IMG_FOLDER+"/"+file_name+".jpg")
    #SHOW IMG RGB CENTER
    # rgb_img[centro_mano[0]][centro_mano[1]]=(0,0,255)

    for border_point in border_points:
        i, j = border_point
        try:
            rgb_img[i+10][j-10]=(0,255,0)
        except:
            continue

    cv2.imshow("iterative", img1)
    cv2.waitKey(0)
    cv2.imshow("IMAGEN",rgb_img)
    cv2.waitKey(0)

    # show_img(data)
    # save_img(data, RESULT_FOLDER + "/" + file_name + "_prev.png")
    # save_img(img1, RESULT_FOLDER + "/" + file_name + ".png")
    # save_img(img, RESULT_FOLDER + "/" + file_name + "_border.png")
    # cv2.imwrite(RESULT_FOLDER + "/" + file_name +"_segmentated.png",rgb_img)
    # # np.savetxt(RESULT_FOLDER + "/" + file_name, img, delimiter=",", fmt='%d')