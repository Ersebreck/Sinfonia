import cv2
import random
#import detect

im = cv2.imread("../Sinfonia/datos/orion.jpg")
cv2.rectangle(im,(213,13),(371,364),[random.randint(0, 255) for _ in range(3)],thickness=1, lineType= cv2.LINE_AA)

cv2.imshow("fotito",im)
cv2.waitKey(5000)
cv2.destroyAllWindows()