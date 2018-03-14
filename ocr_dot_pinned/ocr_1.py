import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import pytesseract

path1 = 'C:\\Users\\JSreekantan\\Desktop\\OCR\\sample4.jpg'
path2 = 'C:\\Users\\JSreekantan\\Desktop\\OCR\\template1.jpg'



# load the image and template
img = cv2.imread(path1)
dot = cv2.imread(path2,0)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(dotheight, dotwidth) = dot.shape[::-1]

#template matching
result = cv2.matchTemplate(img_gray, dot, cv2.TM_CCOEFF_NORMED)
threshold = 0.62
loc = np.where( result >= threshold)

# grab the bounding box of dot and fill them black
for pt in zip(*loc[::-1]):
    topLeft = pt
    botRight = (pt[0] + dotwidth, pt[1] + dotheight)
    img_gray[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = 0

#Blurring using medianblurr
median = cv2.medianBlur(img_gray,5)
#Canny Edge Detection
#img_canny = cv2.Canny(median,250,250)

#Thresholding to binarize
thresh = 0
img_th = cv2.threshold(median, thresh, 255, cv2.THRESH_BINARY)[1]

#image_resizing
img_rs = cv2.resize(img_th,(120, 90), interpolation = cv2.INTER_AREA)


############################################################################################


# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
os.chdir('C:\\Users\\JSreekantan\\Desktop\\OCR')
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, img_rs)


from PIL import Image
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)


# display the images
cv2.imshow("img", img_rs)
#cv2.imshow("dot", dot)
cv2.waitKey(0)