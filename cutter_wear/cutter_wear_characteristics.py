import cv2
import numpy as np
import pandas as pd
import imutils
from matplotlib import pyplot as plt

#path1 = 'D:\\Schlumberger\\cutter_wear\\trail\\almost_circle.png'
#path1 = 'D:\\Schlumberger\\cutter_wear\\trail\\full_circle.png'
#path1 = 'D:\\Schlumberger\\cutter_wear\\trail\\half_circle.png'
#path1 = 'D:\\Schlumberger\\cutter_wear\\trail\\quarter.png'
#path1 = 'D:\\Schlumberger\\cutter_wear\\trail\\three_quarter.png'
path1 = 'D:\\Schlumberger\\cutter_wear\\trail\\side_cut.png'


# Load the image and template
img = cv2.imread(path1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("image",img)
cv2.waitKey(0)

#houg circle transformation
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 100,param1=50,param2=20,minRadius=0,maxRadius=0)
print(circles)
radius = circles[0][0][2]
print(radius)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

img_gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('circles', img_gray1)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()


#Thresholding to binarize
ret, thresh = cv2.threshold(img_gray1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('thresh', thresh)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

#noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

##Segmentation Story
#Sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

#Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)


#Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

cv2.imshow('sure_fg', sure_bg)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

ret, markers = cv2.connectedComponents(sure_bg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv2.imshow('final', img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

#thresholding a color image, here keeping only the blue in the image
th=cv2.inRange(img,(255,0,0),(255,0,0)).astype(np.uint8)


#inverting the image so components become 255 seperated by 0 borders.
th=cv2.bitwise_not(th)

cv2.imshow('outline', th)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

#calling connectedComponentswithStats to get the size of each component
nb_comp,output,sizes,centroids=cv2.connectedComponentsWithStats(th,connectivity=4)

#outputs=cv2.connectedComponentsWithStats(th,connectivity=4)
#taking away the background
nb_comp-=1; sizes=sizes[1:,-1]; centroids=centroids[1:,:]

print(sizes[1])
print(radius)
area = 3.14*radius*radius
wear_range = 1 - (sizes[1]/area)
wear_classification = 10*(round(wear_range,1))

print("The total wear identified is around %.2f and is classified as type %d" %(wear_range,wear_classification))